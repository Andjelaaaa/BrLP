import os, argparse, sys
# 1. Figure out where we live
here = os.path.dirname(__file__)                   # .../scripts/evaluate
scripts_root = os.path.abspath(os.path.join(here, '..'))  # .../scripts
# 2. Add that to sys.path
if scripts_root not in sys.path:
    sys.path.insert(0, scripts_root)
import nibabel as nib
import torch
import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD
from training.train_decoder_pfv import init_autoencoder, AgeConditionalDecoder, FullPredictor
from brlp import const

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True)
    p.add_argument("--dec_ckpt",   required=True)
    p.add_argument("--aekl_ckpt",  required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--fold",       type=int, default=None,
                   help="If set, test set = rows where split == fold.")
    p.add_argument("--all",       type=bool, default=None,
                   help="If to use all elements in csv as test")
    p.add_argument("--split_col", type=str, default="split",
                        help="Column with fold ids (default: 'split').")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) read CSV and pick test rows
    df = pd.read_csv(args.csv)

    if args.fold is not None:
        if "split" not in df.columns:
            raise KeyError(f"'split' column not found in {args.csv}")
        test = df[df["split"] == args.fold].reset_index(drop=True)
        sel_desc = f"split == {args.fold}"
        if args.all:
            test = df
    else:
        # fallback to legacy behavior if no fold provided
        if "split" not in df.columns:
            raise RuntimeError("No --fold provided and no 'split' column present.")
        test = df[df["split"] == "test"].reset_index(drop=True)
        sel_desc = "split == 'test'"

    if test.empty:
        raise RuntimeError(f"No rows matched selection: {sel_desc}")

    print(f"[predict] Using {len(test)} samples ({sel_desc}).")

    # 1) rebuild models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = init_autoencoder(args.aekl_ckpt).to(device)
    dec = AgeConditionalDecoder(ae, age_embed_dim=16).to(device)
    # (build projection, load dec_ckpt, wrap in FullPredictor...)
    with torch.no_grad():
        dummy = torch.zeros(1,1,*const.INPUT_SHAPE_AE,device=device)
        mu,sigma = ae.encode(dummy)
        z = ae.sampling(mu,sigma)
    dec.initialize_projection(z.shape,device)
    dec.load_state_dict(torch.load(args.dec_ckpt,map_location=device))
    dec.eval()
    model = FullPredictor(ae, dec).to(device)
    model.eval()

    # 2) build preprocess pipelines
    img_pipe = Compose([
      LoadImageD(keys=['img'], image_only=True),
      EnsureChannelFirstD(keys=['img']),
      SpacingD(pixdim=const.RESOLUTION, keys=['img']),
      ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['img']),
      ScaleIntensityD(minv=0, maxv=1, keys=['img'])
    ])

    seg_pipe = Compose([
        LoadImageD(keys=['seg'], image_only=True),
        EnsureChannelFirstD(keys=['seg']),
        SpacingD(pixdim=const.RESOLUTION, keys=['seg'], mode="nearest"),
        ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE,
                                mode='minimum',
                                keys=['seg']),
    ])
    print('Starting predicting...')
    for _, row in test.iterrows():
        sid = row.subject_id
        T1 = row.starting_image_uid
        T2 = row.followup_image_uid
        # -- predict T2 from T1 image + age
        arr = img_pipe({'img': row.starting_image_path})['img']  # [C,D,H,W]
        x0 = arr.unsqueeze(0).to(device)
        age = torch.tensor([[(row.followup_age)]],device=device)
        with torch.no_grad():
            pred = model(x0, age)[0,0].cpu().numpy()
        nib.save(nib.Nifti1Image(pred, const.MNI152_1P5MM_AFFINE),
                 os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz"))

        # -- also preprocess & save the T2 *ground-truth* segmentation
        seg_arr = seg_pipe({'seg': row.followup_segm_path})['seg']
        # cast to int32
        seg_arr = seg_arr.astype(np.int32)
        nib.save(nib.Nifti1Image(seg_arr[0], const.MNI152_1P5MM_AFFINE),
                 os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz"))

# import os
# import sys
# import subprocess
# import argparse
# from datetime import datetime
# from SynthSeg.predict import predict_segmentation
# import shutil
# import nibabel as nib
# import torch
# import pandas as pd
# import numpy as np
# import wandb
# from monai.metrics import DiceMetric
# from typing import Optional
# from PIL import Image, ImageDraw
# from torch.utils.data import DataLoader

# from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD

# from brlp import (
#     KLDivergenceLoss, GradientAccumulation,
#     init_autoencoder, init_patch_discriminator
# )
# # 2) import exactly what you need from your training script
# from train_decoder_pfv import (
#     init_autoencoder,
#     init_patch_discriminator,
#     AgeConditionalDecoder,
#     FullPredictor,
#     const
# )

# # 1) make sure Python can see your training module
# THIS = os.path.dirname(__file__)
# ROOT = os.path.abspath(os.path.join(THIS, ".."))          # ../evaluate/..
# TRAIN = os.path.join(ROOT, "training")                    # ../training
# sys.path.insert(0, TRAIN)

# def synthseg_and_load(nii_in, out_dir):
#     out = os.path.join(out_dir, os.path.basename(nii_in).replace('.nii.gz','_seg.nii.gz'))
#     subprocess.run([
#         'mri_synthseg',
#         '--i', nii_in,
#         '--o', out
#     ], check=True)
#     return nib.load(out).get_fdata().astype(np.int32)

# def compute_dice_per_label(pred, true):
#     labels = np.unique(true)
#     dice = {}
#     for lbl in labels:
#         if lbl == 0: continue
#         p = (pred==lbl); t = (true==lbl)
#         inter = (p & t).sum()
#         if p.sum()+t.sum()>0:
#             dice[lbl] = 2*inter/(p.sum()+t.sum())
#     return dice

# def log_full_volume_segmentation_gif(
#     true_mask_np: np.ndarray,
#     pred_mask_np: np.ndarray,
#     output_dir: str,
#     tag: str = "test/segmentation_comparison",
#     step: Optional[int] = None,
#     duration: int = 100
# ):
#     """
#     Turn two 3D masks into a side-by-side, slice-by-slice GIF and log to wandb.

#     Args:
#         true_mask_np:   3D array [D,H,W] of ground-truth labels
#         pred_mask_np:   3D array [D,H,W] of predicted labels
#         output_dir:     where to write the temporary GIF
#         tag:            wandb key under which the GIF will be logged
#         step:           optional wandb step index
#         duration:       ms per frame in the GIF
#     """
#     D, H, W = true_mask_np.shape
#     frames = []

#     # normalize each slice to 0–255 for PIL
#     t_max = float(true_mask_np.max() or 1)
#     p_max = float(pred_mask_np.max() or 1)

#     for z in range(D):
#         # Extract and scale to 0–255
#         t_slice = (true_mask_np[z].astype(np.float32) / t_max * 255).astype(np.uint8)
#         p_slice = (pred_mask_np[z].astype(np.float32) / p_max * 255).astype(np.uint8)

#         # Make PIL images and annotate
#         im_t = Image.fromarray(t_slice, mode="L").convert("RGB")
#         im_p = Image.fromarray(p_slice, mode="L").convert("RGB")
#         draw = ImageDraw.Draw(im_t); draw.text((5,5), "TRUE", fill="white")
#         draw = ImageDraw.Draw(im_p); draw.text((5,5), "PRED", fill="white")

#         # Composite side by side
#         combo = Image.new("RGB", (W*2, H))
#         combo.paste(im_t, (0,0))
#         combo.paste(im_p, (W,0))
#         frames.append(combo)

#     # Write out GIF
#     os.makedirs(output_dir, exist_ok=True)
#     gif_path = os.path.join(output_dir, "seg_comparison_fullvol.gif")
#     frames[0].save(
#         gif_path,
#         format="GIF",
#         save_all=True,
#         append_images=frames[1:],
#         duration=duration,
#         loop=0
#     )

#     # Log to wandb
#     if step is not None:
#         wandb.log({tag: wandb.Video(gif_path, format="gif")}, step=step)
#     else:
#         wandb.log({tag: wandb.Video(gif_path, format="gif")})

#     # Cleanup
#     os.remove(gif_path)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_csv',    required=True, type=str,
#                         help="CSV with columns including: split, starting_image_path, followup_image_path, starting_age, followup_age")
#     parser.add_argument('--output_dir',     required=True, type=str,
#                         help="Where to save predictions/Dice scores")
#     parser.add_argument('--aekl_ckpt',      required=True, type=str,
#                         help="Path to pretrained AutoencoderKL checkpoint")
#     parser.add_argument('--dec_ckpt',       required=True, type=str,
#                         help="Path to your trained conditional‐decoder .pth")
#     parser.add_argument('--project',        default="age_conditional_decoder_eval", type=str,
#                         help="W&B project name")
#     parser.add_argument('--run_name',       default=None, type=str,
#                         help="W&B run name (timestamp if omitted)")
#     parser.add_argument('--mode',           choices=['save_nifti','dice'], default='dice',
#                         help="Whether to save predicted volumes or compute Dice via SynthSeg")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     # reproducibility + device
#     torch.manual_seed(0)
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # set up W&B
#     run_name = args.run_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")
#     wandb.init(project=args.project, name=run_name, config=vars(args), mode="offline")

#     # ―― Rebuild models ――
#     autoencoder  = init_autoencoder(args.aekl_ckpt).to(DEVICE)
#     cond_decoder = AgeConditionalDecoder(autoencoder, age_embed_dim=16).to(DEVICE)

#     # build its projection (dummy forward)
#     with torch.no_grad():
#         dummy = torch.zeros(1,1,*const.INPUT_SHAPE_AE, device=DEVICE)
#         mu, sigma = autoencoder.encode(dummy)
#         z = autoencoder.sampling(mu, sigma)
#     cond_decoder.initialize_projection(z.shape, DEVICE)

#     # load your trained decoder weights
#     cond_decoder.load_state_dict(torch.load(args.dec_ckpt, map_location=DEVICE))
#     cond_decoder.eval()

#     # wrap in FullPredictor
#     model = FullPredictor(autoencoder, cond_decoder).to(DEVICE)
#     model.eval()

#     # ―― Load test split CSV ――
#     df = pd.read_csv(args.dataset_csv)
#     test_df = df[df.split=='test'].reset_index(drop=True)

#     results = []
#     for idx, row in test_df.iterrows():
#         subj = row.subject_id
#         t1_path = row.starting_image_path
#         t2_path = row.followup_image_path
#         # normalize age
#         age1 = torch.tensor([[(row.followup_age - 1.0)/6.0]], device=DEVICE)

#         # ― load + preprocess T1 exactly as in training
#         pipeline = Compose([
#             LoadImageD(keys=['img'], image_only=True),
#             EnsureChannelFirstD(keys=['img']),
#             SpacingD(pixdim=const.RESOLUTION, keys=['img']),
#             ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['img']),
#             ScaleIntensityD(minv=0.0, maxv=1.0, keys=['img'])
#         ])
#         img1_np = pipeline({'img': t1_path})['img']  # numpy [C,D,H,W]
#         x0 = torch.from_numpy(img1_np[None]).to(DEVICE)  # [1,1,D,H,W]

#         # ― run model
#         with torch.no_grad():
#             x2_pred = model(x0, age1)  # [1,1,D,H,W]
#         pred_vol = x2_pred[0,0].cpu().numpy()

#         # ― save NIfTI
#         pred_nii = nib.Nifti1Image(pred_vol, affine=np.eye(4))
#         out_pred = os.path.join(args.output_dir, f"{subj}_pred.nii.gz")
#         nib.save(pred_nii, out_pred)

#         if args.mode == 'dice':
#             # ── 1) run SynthSeg on your *predicted* T2
#             seg_pred = predict_segmentation(
#                 out_pred,
#                 model_specs=args.synthseg_model,
#                 output_dir=args.output_dir
#             )

#             # ── 2) load the *true* T2 segmentation directly from CSV
#             seg_pipe = Compose([
#                 LoadImageD(keys=['seg'], image_only=True),
#                 EnsureChannelFirstD(keys=['seg']),
#                 SpacingD(pixdim=const.RESOLUTION, keys=['seg']),
#                 ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE,
#                                      mode='minimum',
#                                      keys=['seg']),
#             ])
#             seg_true_np = seg_pipe({'seg': row.followup_segm_path})['seg']
#             # save it temporarily so DiceMetric can read it via nibabel
#             tmp_true = os.path.join(args.output_dir, f"{subj}_trueSeg.nii.gz")
#             nib.save(nib.Nifti1Image(seg_true_np.astype(np.int32),
#                                      np.eye(4)), tmp_true)
            
#             pred_mask_np = nib.load(seg_pred).get_fdata().astype(np.int32)  # [D,H,W]
#             true_mask_np = seg_true_np                                      # [D,H,W]

#             # Compute per-label Dice 
#             dice_metric = DiceMetric(include_background=False, reduction="mean")
#             p = nib.load(seg_pred).get_fdata().astype(np.int32)[None]
#             t = nib.load(tmp_true).get_fdata().astype(np.int32)[None]
#             labels = np.unique(t)[1:]
#             dice_scores = {}    
#             for L in labels:
#                 onep = torch.from_numpy((p==L).astype(np.float32))[None]
#                 onet = torch.from_numpy((t==L).astype(np.float32))[None]
#                 dice = dice_metric(onep, onet).item()
#                 dice_scores[int(L)] = dice

#             # log & record
#             wandb.log({f"test/dice_label_{L}": d for L,d in dice_scores.items()}, step=idx)
#             results.append({"subject_id": subj, **{f"dice_{L}": d for L,d in dice_scores.items()}})

#             # Log the full-volume side-by-side GIF
#             log_full_volume_segmentation_gif(
#                 true_mask_np=true_mask_np,
#                 pred_mask_np=pred_mask_np,
#                 output_dir=args.output_dir,
#                 tag="Eval/Segmentation_FullVolume",
#                 step=idx,
#                 duration=150
#             )

#             # clean up our temporary true-seg file
#             os.remove(tmp_true)

#     # save a summary CSV
#     if args.mode == 'dice':
#         pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "test_dice_scores.csv"), index=False)
#         wandb.save(os.path.join(args.output_dir, "test_dice_scores.csv"))

#     wandb.finish()
