import os
import re
import csv
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.nn import L1Loss
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from monai import transforms
from brlp import const, get_dataset_from_pd, networks

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INT32_MAX = 2_147_483_647

# === MONAI TRANSFORMS ===
def get_transforms():
    return transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

# === DATASET LOADING ===
def get_loader(csv_path, cache_dir, test_fold, seed, batch_size, num_workers):
    set_determinism(seed)
    df = pd.read_csv(csv_path)

    # Debug info
    print("[data] rows:", len(df))
    if "split" in df.columns:
        print("[data] split unique (first 20):", sorted(df["split"].astype(str).unique())[:20])

    # --- Selection logic ---
    if "split" not in df.columns:
        raise ValueError("CSV must contain a 'split' column when using fold-based selection.")

    s = df["split"]

    # Case A: split is a numeric fold id (your case)
    if pd.api.types.is_numeric_dtype(s) or s.astype(str).str.fullmatch(r"\d+").all():
        mask = (s.astype(int) == int(test_fold))

    test_df = df[mask].copy()
    print(f"[data] fold={test_fold} seed={seed}: selected {len(test_df)} test rows")

    if test_df.empty:
        print(f"[warn] No test data for fold {test_fold}.")
        return None

    testset = get_dataset_from_pd(test_df, get_transforms(), cache_dir)
    print(f"[data] dataset size={len(testset)}; batch_size={batch_size}")
    return DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

# === LOSS FUNCTION ===
loss_fn = L1Loss()

# === MODEL EVALUATION ===
def _tensor_info(t, name="tensor"):
    try:
        # min/max can fail on empty tensors; guard it
        _min = t.min().item() if t.numel() > 0 else float("nan")
        _max = t.max().item() if t.numel() > 0 else float("nan")
    except Exception:
        _min, _max = float("nan"), float("nan")
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"numel={t.numel():,}, min={_min:.3g}, max={_max:.3g}")

# === MODEL EVALUATION ===
def evaluate_model(model, loader, ckpt_name="(unknown)"):
    if loader is None:
        return None
    model.eval()
    total_loss, count = 0.0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {ckpt_name}", leave=False), start=1):
            images = batch["image"].to(DEVICE, non_blocking=True)

            # Always show input stats for the first batch
            if batch_idx == 1:
                print("[debug]", _tensor_info(images, "images"))
                try:
                    from brlp import const
                    print(f"[debug] const.INPUT_SHAPE_AE={getattr(const, 'INPUT_SHAPE_AE', None)} "
                          f"const.RESOLUTION={getattr(const, 'RESOLUTION', None)}")
                except Exception:
                    pass

            # Guard: input too large for int32 indexing
            if images.numel() > INT32_MAX:
                print(f"⚠️  [{ckpt_name}] images too large for int32 indexing: numel={images.numel():,} (> {INT32_MAX:,})")
                return None  # skip this checkpoint cleanly

            try:
                recon, _, _ = model(images)
            except RuntimeError as e:
                # Print diagnostics and skip this checkpoint
                print(f"⚠️  [{ckpt_name}] forward() failed on batch {batch_idx} with RuntimeError: {e}")
                print("[debug]", _tensor_info(images, "images"))
                return None
            except Exception as e:
                print(f"⚠️  [{ckpt_name}] forward() failed on batch {batch_idx}: {type(e).__name__}: {e}")
                print("[debug]", _tensor_info(images, "images"))
                return None

            # Output stats
            if batch_idx == 1:
                print("[debug]", _tensor_info(recon, "recon"))

            # Guard: output too large
            if recon.numel() > INT32_MAX:
                print(f"⚠️  [{ckpt_name}] recon too large for int32 indexing: numel={recon.numel():,} (> {INT32_MAX:,})")
                print("[debug]", _tensor_info(recon, "recon"))
                return None

            # Guard: shape mismatch (very common cause of absurd upsamples)
            if recon.shape != images.shape:
                print(f"⚠️  [{ckpt_name}] shape mismatch: recon {tuple(recon.shape)} vs images {tuple(images.shape)}")
                print("[hint] Check const.INPUT_SHAPE_AE, const.RESOLUTION, Resize/Spacing, and checkpoint architecture.")
                return None

            try:
                loss = loss_fn(recon, images)
            except RuntimeError as e:
                print(f"⚠️  [{ckpt_name}] loss computation failed on batch {batch_idx}: {e}")
                print("[debug]", _tensor_info(images, "images"))
                print("[debug]", _tensor_info(recon, "recon"))
                return None

            total_loss += loss.item() * images.size(0)
            count += images.size(0)

    if count == 0:
        print(f"[warn] No batches seen for {ckpt_name}.")
        return None
    return total_loss / count

# === MAIN ===
def main(args):
    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for fold_name in sorted(os.listdir(args.ckpt_root_dir)):
        match = re.match(r'fold_(\d+)', fold_name)
        if not match:
            continue
        fold = int(match.group(1))

        if fold < 2:
            print('Skipping fold1')
            continue
        ckpt_dir = os.path.join(args.ckpt_root_dir, fold_name)
        cache_dir = os.path.join(args.cache_dir, fold_name)

        print(f"\n🧪 Evaluating fold {fold} using test set test{fold}...")

        for seed in seeds:
            print(f"\n🌱 Seed {seed}")
            test_loader = get_loader(args.csv_path, cache_dir, fold, seed,
                                    args.batch_size, args.num_workers)
            if test_loader is None:
                continue

            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth") and "autoencoder" in f])
            print(f"[eval] fold={fold} seed={seed}: {len(ckpts)} checkpoints to evaluate")

            seed_results = []
            for i, fname in enumerate(ckpts, 1):
                ckpt_path = os.path.join(ckpt_dir, fname)
                print(f"[{i}/{len(ckpts)}] 📂 Loading {fname}")
                try:
                    model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
                    val_loss = evaluate_model(model, test_loader, ckpt_name=fname)
                    if val_loss is None:
                        print(f"   ↳ skipped (no data)")
                        continue
                    print(f"   ↳ test L1 loss = {val_loss:.6f}")
                    seed_results.append((fold, seed, fname, val_loss))
                except Exception as e:
                    print(f"⚠️ Failed on {fname}: {type(e).__name__}: {e}")

            if seed_results:
                best_model = min(seed_results, key=lambda x: x[3])
                print(f"\n✅ Best model for fold {fold}, seed {seed}: {best_model[2]} - L1 = {best_model[3]:.6f}")
                all_results.append(best_model)
            else:
                print(f"[warn] No results collected for fold={fold}, seed={seed}.")

    # === Compute best epoch across seeds for each fold ===
    best_per_fold = {}
    # first, parse epoch numbers and accumulate
    fold_epoch_losses = {}  # (fold, epoch) -> [losses]
    for fold, seed, fname, loss in all_results:
        m = re.search(r'epoch_(\d+)', fname)
        if not m:
            continue
        epoch = int(m.group(1))
        fold_epoch_losses.setdefault((fold, epoch), []).append(loss)

    # compute means and pick best epoch per fold
    for (fold, epoch), losses in fold_epoch_losses.items():
        mean_loss = sum(losses) / len(losses)
        if (fold not in best_per_fold) or (mean_loss < best_per_fold[fold][1]):
            best_per_fold[fold] = (epoch, mean_loss)

    # === Save summary CSV ===
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "mean_test_loss"])
        if not best_per_fold:
            print("[warn] No best-per-fold results to write. Check your data filtering above.")
        for fold in sorted(best_per_fold):
            epoch, loss = best_per_fold[fold]
            writer.writerow([fold, epoch, f"{loss:.6f}"])
    print(f"\n📄 Best-epoch-per-fold summary saved to: {args.output_csv}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True,
                        help="Path to dataset CSV file")
    parser.add_argument("--cache_dir", required=True,
                        help="Path to root cache directory (containing foldX subfolders)")
    parser.add_argument("--ckpt_root_dir", required=True,
                        help="Path to root checkpoint directory (containing foldX subfolders)")
    parser.add_argument("--output_csv", default="test_losses.csv",
                        help="Where to save output CSV")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated list of seeds")

    args = parser.parse_args()
    main(args)
