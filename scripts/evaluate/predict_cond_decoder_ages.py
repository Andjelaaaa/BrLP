import os, argparse, sys
import nibabel as nib
import torch
import numpy as np
import pandas as pd

from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, SpacingD,
    ResizeWithPadOrCropD, ScaleIntensityD
)

# adjust imports to match your repo
from training.train_decoder_ages import init_autoencoder, AgeVectorConditionalDecoder, FullPredictor
from brlp import const


# def age_months_to_norm(months: float, age_min_m: float = 12.0, age_max_m: float = 84.0) -> float:
#     return (float(months) - age_min_m) / (age_max_m - age_min_m)


def load_cond_decoder_ckpt(cond_decoder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ["cond_decoder", "cond_decoder_state_dict", "state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                cond_decoder.load_state_dict(ckpt[k], strict=True)
                return
        if any(torch.is_tensor(v) for v in ckpt.values()):
            cond_decoder.load_state_dict(ckpt, strict=True)
            return
        raise RuntimeError(f"Unrecognized checkpoint dict keys: {list(ckpt.keys())}")
    cond_decoder.load_state_dict(ckpt, strict=True)


def build_pipelines():
    img_pipe = Compose([
        LoadImageD(keys=["img"], image_only=True),
        EnsureChannelFirstD(keys=["img"]),
        SpacingD(pixdim=const.RESOLUTION, keys=["img"]),
        ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["img"]),
        ScaleIntensityD(minv=0, maxv=1, keys=["img"]),
    ])

    seg_pipe = Compose([
        LoadImageD(keys=["seg"], image_only=True),
        EnsureChannelFirstD(keys=["seg"]),
        SpacingD(pixdim=const.RESOLUTION, keys=["seg"], mode="nearest"),
        ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["seg"]),
    ])
    return img_pipe, seg_pipe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True, help="all_pred_pairs_with_cv_splits.csv")
    p.add_argument("--fold",       type=int, required=True, help="Fold id (1..5) to evaluate")
    p.add_argument("--split_col",  type=str, default="split", help="Column name for split ('train'/'val'/'test')")
    p.add_argument("--fold_col",   type=str, default="fold",   help="Column name for fold id")

    p.add_argument("--dec_ckpt",   required=True, help="Conditional decoder checkpoint for THIS fold")
    p.add_argument("--aekl_ckpt",  required=True, help="Autoencoder checkpoint for THIS fold")
    p.add_argument("--age_embed_dim", type=int, default=16)

    p.add_argument("--out_dir",    required=True)
    p.add_argument("--save_gt_seg", action="store_true")
    p.add_argument("--save_gt_t2",  action="store_true")

    # age normalization (must match training!)
    p.add_argument("--age_min_months", type=float, default=12.0)
    p.add_argument("--age_max_months", type=float, default=84.0)

    # optional: run on all rows of that fold (ignores split == test)
    p.add_argument("--all_rows_in_fold", action="store_true")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # select test rows for this fold
    if args.fold_col not in df.columns:
        raise KeyError(f"Missing fold column '{args.fold_col}' in {args.csv}")
    if args.split_col not in df.columns and not args.all_rows_in_fold:
        raise KeyError(f"Missing split column '{args.split_col}' in {args.csv}")

    if args.all_rows_in_fold:
        test = df[df[args.fold_col] == args.fold].reset_index(drop=True)
        sel_desc = f"{args.fold_col} == {args.fold} (ALL rows)"
    else:
        test = df[(df[args.fold_col] == args.fold) & (df[args.split_col] == "test")].reset_index(drop=True)
        sel_desc = f"{args.fold_col} == {args.fold} AND {args.split_col} == 'test'"

    if test.empty:
        raise RuntimeError(f"No rows matched selection: {sel_desc}")
    print(f"[predict] Using {len(test)} samples ({sel_desc}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build models ----
    ae = init_autoencoder(args.aekl_ckpt).to(device)
    ae.eval()
    for p_ in ae.parameters():
        p_.requires_grad = False

    cond_decoder = AgeVectorConditionalDecoder(pretrained_ae=ae, age_embed_dim=args.age_embed_dim).to(device)

    # build projection shape EXACTLY like training
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *const.INPUT_SHAPE_AE, device=device)
        mu, sigma = ae.encode(dummy)
        z = ae.sampling(mu, sigma)
    cond_decoder.initialize_projection(z.shape, device)

    load_cond_decoder_ckpt(cond_decoder, args.dec_ckpt, device)
    cond_decoder.eval()

    model = FullPredictor(ae, cond_decoder).to(device)
    model.eval()

    img_pipe, seg_pipe = build_pipelines()

    # ---- inference ----
    manifest_rows = []
    print("[predict] Starting inference...")

    for i, row in test.iterrows():
        sid = row["subject_id"]
        T1  = row["starting_image_uid"]
        T2  = row["followup_image_uid"]

        # baseline image
        x0 = img_pipe({"img": row["starting_image_path"]})["img"]  # [C,D,H,W]
        x0 = x0.unsqueeze(0).to(device)                            # [1,C,D,H,W]

        # ages (months -> normalized like training)
        a0_mo = float(row["starting_age_bef_norm"])
        a1_mo = float(row["followup_age_bef_norm"])
        a0_norm = float(row["starting_age"])
        a1_norm = float(row["followup_age"])
        a0 = torch.tensor([[age_months_to_norm(a0_norm, args.age_min_months, args.age_max_months)]], device=device)
        a1 = torch.tensor([[age_months_to_norm(a1_norm, args.age_min_months, args.age_max_months)]], device=device)

        with torch.no_grad():
            pred = model(x0, a0, a1)[0, 0].detach().cpu().numpy()

        pred_path = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2pred.nii.gz")
        nib.save(nib.Nifti1Image(pred.astype(np.float32), const.MNI152_1P5MM_AFFINE), pred_path)

        # No need to convert segm_path to AE space everything will be done in SynthSeg 1mm space
        # seg_path_out = None
        # if args.save_gt_seg and pd.notna(row.get("followup_segm_path", None)):
        #     seg = seg_pipe({"seg": row["followup_segm_path"]})["seg"]  # [C,D,H,W]
        #     seg = seg[0].detach().cpu().numpy().astype(np.int32)
        #     seg_path_out = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2seg_gt_AEspace.nii.gz")
        #     nib.save(nib.Nifti1Image(seg, const.MNI152_1P5MM_AFFINE), seg_path_out)

        t2_path_out = None
        if args.save_gt_t2 and pd.notna(row.get("followup_image_path", None)):
            t2 = img_pipe({"img": row["followup_image_path"]})["img"]
            t2 = t2[0].detach().cpu().numpy().astype(np.float32)
            t2_path_out = os.path.join(args.out_dir, f"{sid}_{T1}_to_{T2}_T2_gt_AEspace.nii.gz")
            nib.save(nib.Nifti1Image(t2, const.MNI152_1P5MM_AFFINE), t2_path_out)

        manifest_rows.append({
            "fold": args.fold,
            "subject_id": sid,
            "starting_image_uid": T1,
            "followup_image_uid": T2,
            "pred_followup_path": pred_path,
            "starting_img_path": row["starting_image_path"],
            "starting_segm_path": row["starting_segm_path"],
            "gt_followup_segm_path": row["followup_segm_path"],
            "gt_followup_path_AEspace": t2_path_out,
            "starting_age_months": a0_mo,
            "followup_age_months": a1_mo,
            "sex": row["sex"]
        })

        if (i + 1) % 25 == 0:
            print(f"[predict] {i+1}/{len(test)} done")

    man_path = os.path.join(args.out_dir, "pred_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(man_path, index=False)
    print(f"[predict] Wrote manifest: {man_path}")
