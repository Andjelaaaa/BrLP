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
from monai.transforms import Compose, LoadImageD, EnsureTyped, EnsureChannelFirstD, SpacingD, Invertd, ResizeWithPadOrCropD, ScaleIntensityD
from training.train_decoder_pfv import init_autoencoder, AgeConditionalDecoder, FullPredictor
from brlp import const
from monai.data.meta_tensor import MetaTensor

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
      EnsureTyped(keys=['img'], track_meta=True),
      SpacingD(pixdim=const.RESOLUTION, keys=['img']),
      ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['img']),
      ScaleIntensityD(minv=0, maxv=1, keys=['img'])
    ])

    # ---- Spatial-only pipe: used just to record transform history for inversion
    spatial_pipe = Compose([
        LoadImageD(keys=['img'], image_only=False),          # keep meta!
        EnsureChannelFirstD(keys=['img']),
        EnsureTyped(keys=['img'], track_meta=True),          # MetaTensor with _transforms
        SpacingD(keys=['img'], pixdim=const.RESOLUTION),# mode='bilinear'),
        ResizeWithPadOrCropD(keys=['img'], spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
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

        ### WAY 1: Do all in the prediction space
        # nib.save(nib.Nifti1Image(pred, const.MNI152_1P5MM_AFFINE),
        #          os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz"))

        # # -- also preprocess & save the T2 *ground-truth* segmentation
        # seg_arr = seg_pipe({'seg': row.followup_segm_path})['seg']
        # # cast to int32
        # seg_arr = seg_arr.astype(np.int32)
        # nib.save(nib.Nifti1Image(seg_arr[0], const.MNI152_1P5MM_AFFINE),
        #          os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2segtform.nii.gz"))

        ### WAY 2: Do all in the initial space
        # --- 3) ALSO preprocess the follow-up T2 (just to capture its transform record)
        # We won't use the image tensor—only its meta and transform history for inversion.
        # -- run spatial pipe on follow-up T2 to capture forward transform history
        t2_dict =  spatial_pipe({'img': row.followup_image_path})    # t2d['img'] is MetaTensor [1,D,H,W]


        # --- 4) prepare a dict that contains the prediction and the T2 meta
        # Make pred a MetaTensor with an affine/grid representing the network space.
        pred_meta = t2_dict['img'].meta.copy()        # clone meta structure
        # BUT critically: pred is currently in *network space* (after img_pipe).
        # So we want to attach the *same* meta as t2_dict['img'] currently has,
        # then invert using the exact transforms that produced t2_dict['img'].
        pred_t = torch.as_tensor(pred)
        pred_mt = MetaTensor(pred_t.unsqueeze(0), meta=pred_meta) 

        data_for_inverse = {
            'img': t2_dict['img'],           # carries the forward history for T2
            'pred': pred_mt                  # to be inverted back to native T2 space
        }

        # --- 5) invert using the forward transform graph of `spatial_pipe` applied to 'img'
        invert_pred = Invertd(
            keys=['pred'],
            transform=spatial_pipe,
            orig_keys=['img'],               # use the history recorded for 'img'
            nearest_interp=False,            # linear for MRI intensities
            to_tensor=True,
            device=None
        )

        out_native = invert_pred(data_for_inverse)
        pred_native  = out_native['pred'][0]       # drop channel -> [D,H,W]
        affine_native = pred_native.affine   # original follow-up T2 affine

        ref_img = nib.load(row.followup_image_path)
        ref_aff = ref_img.affine
        ref_hdr = ref_img.header.copy()
        print("ref header:",        ref_hdr)
        print("shapes:", pred_native.shape, nib.load(row.followup_image_path).shape)

        out = nib.Nifti1Image(pred_native.numpy(), ref_aff, header=ref_hdr)
        out.set_qform(ref_aff, code=1)   # code=1: "scanner"
        out.set_sform(ref_aff, code=1)

        # --- 6) save aligned to the *original* follow-up T2 header
        # Use the original affine stored in meta:
        nib.save(
            out,
            os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz")
        )

        saved = nib.load(os.path.join(args.out_dir, f"{sid}_{T1}_{T2}_T2pred.nii.gz"))
        print("SAVED zooms:", saved.header.get_zooms()[:3])             # -> (1.0, 1.0, 1.0)
        print("SAVED codes:", int(saved.header['qform_code']),
                            int(saved.header['sform_code'])) 
