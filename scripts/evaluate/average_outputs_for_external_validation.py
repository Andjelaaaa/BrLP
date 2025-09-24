#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/andim/scratch/brlp/predict-cond-decoder-imgs/external")  # has fold_1..fold_5
OUT  = ROOT / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

# discover fold dirs (supports "fold_1" or "fold1")
fold_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("fold")])
if not fold_dirs:
    raise SystemExit(f"No fold_* directories under {ROOT}")

# group predictions by basename (same filename across folds)
name_to_paths = defaultdict(list)
for fd in fold_dirs:
    for p in fd.glob("*_T2pred.nii.gz"):
        name_to_paths[p.name].append(p)

n_folds = len(fold_dirs)
kept = {name: paths for name, paths in name_to_paths.items() if len(paths) == n_folds}
missing = {name: paths for name, paths in name_to_paths.items() if len(paths) != n_folds}
print(f"Folds found: {n_folds}")
print(f"Subjects with complete {n_folds}/" + f"{n_folds} folds: {len(kept)}")
if missing:
    print(f"WARNING: {len(missing)} subjects missing some folds (will be skipped).")

def save_float32(img_like, data, out_path):
    hdr = img_like.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32, copy=False), img_like.affine, hdr), out_path)

for name, paths in sorted(kept.items()):
    # load all fold predictions
    imgs  = [nib.load(str(p)) for p in paths]
    datas = [im.get_fdata(dtype=np.float32) for im in imgs]

    # sanity checks
    shapes = {d.shape for d in datas}
    if len(shapes) != 1:
        print(f"SKIP (shape mismatch) {name}: {shapes}")
        continue
    aff0 = imgs[0].affine
    if not all(np.allclose(aff0, im.affine) for im in imgs[1:]):
        print(f"SKIP (affine mismatch) {name}")
        continue

    # ensemble stats
    stack = np.stack(datas, axis=0)           # [K, D, H, W]
    mean  = stack.mean(axis=0)
    med   = np.median(stack, axis=0)
    var   = stack.var(axis=0)                 # population var; use ddof=1 for sample

    # (optional) clip to [0,1] since inputs were scaled
    mean = np.clip(mean, 0.0, 1.0)
    med  = np.clip(med,  0.0, 1.0)

    # output filenames
    base = name[:-len("_T2pred.nii.gz")]  # strip suffix
    out_mean = OUT / f"{base}_ens-mean.nii.gz"
    out_median = OUT / f"{base}_ens-median.nii.gz"
    out_var = OUT / f"{base}_ens-var.nii.gz"

    # save (use first image’s affine/header as template)
    save_float32(imgs[0], mean, out_mean)
    save_float32(imgs[0], med,  out_median)
    save_float32(imgs[0], var,  out_var)

    print(f"OK {base} → {out_mean.name}, {out_median.name}, {out_var.name}")
