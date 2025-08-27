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

        if fold < 5:
            print('Skipping folds')
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


"""
This script evaluates autoencoder checkpoints across folds and seeds, using either
per‐fold test splits or an external dataset. For each fold it:

  1. Loads one or more evaluation seeds:
       - If --seed is provided, evaluates only that single seed.
       - Otherwise, parses --seeds as a comma‐separated list.
  2. Loads checkpoints from the specified root directory:
       ckpt_root_dir/fold_X/*.pth
     Only files containing "autoencoder" are included.
  3. Prepares test data:
       - By default, selects rows from --csv_path where split == fold index.
       - If --external is set, ignores "split" and evaluates on the full CSV.
     Data are cached under --cache_dir/fold_X for efficiency.
  4. Applies MONAI transforms to images:
       • Load and convert to channel‐first format
       • Resample to const.RESOLUTION
       • Resize/pad/crop to const.INPUT_SHAPE_AE
       • Scale intensities to [0,1]
  5. Evaluates each checkpoint:
       - Performs forward pass through the model
       - Computes reconstruction L1 loss against the input
       - Logs shape mismatches or other errors and skips failing checkpoints
  6. Collects results:
       - Per checkpoint: (fold, seed, checkpoint filename, test loss)
       - Per fold: finds the epoch with lowest mean test loss across seeds
  7. Writes outputs:
       - A summary CSV (--output_csv) with the best epoch per fold
       - Optionally, a per‐checkpoint CSV (--per_ckpt_csv) with all results

Arguments:
    --csv_path       Path to dataset CSV (must contain "split" unless --external)
    --cache_dir      Directory for cached preprocessed images (per fold subdirs)
    --ckpt_root_dir  Root directory of checkpoints, with fold_* subfolders
    --output_csv     Path to summary CSV of best epochs per fold (default: test_losses_summary.csv)
    --per_ckpt_csv   Optional path to save full per‐checkpoint results
    --batch_size     DataLoader batch size (default: 2)
    --num_workers    Number of DataLoader workers (default: 4)
    --external       Evaluate on the full CSV instead of per‐fold splits
    --seed           Single seed for evaluation (overrides --seeds)
    --seeds          Comma‐separated list of seeds (default: 0,1,2,3,4)

Outputs:
    1. A summary CSV (--output_csv) with columns:
         • fold         — fold index (from checkpoint folder name)
         • best_epoch   — epoch number with lowest mean loss across seeds
         • mean_test_loss_across_seeds — mean L1 loss for that epoch
    2. (Optional) A per‐checkpoint CSV (--per_ckpt_csv) with columns:
         • fold         — fold index
         • seed         — evaluation seed
         • checkpoint   — checkpoint filename
         • test_loss_L1 — L1 loss on test set

Authors:
    Andjela Dimitrijevic
"""


# import os
# import re
# import csv
# import torch
# import argparse
# import pandas as pd
# from tqdm import tqdm
# from torch.nn import L1Loss
# from monai.utils import set_determinism
# from torch.utils.data import DataLoader

# from monai import transforms
# from brlp import const, get_dataset_from_pd, networks

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# INT32_MAX = 2_147_483_647


# # === MONAI TRANSFORMS ===
# def get_transforms():
#     return transforms.Compose([
#         transforms.CopyItemsD(keys={'image_path'}, names=['image']),
#         transforms.LoadImageD(image_only=True, keys=['image']),
#         transforms.EnsureChannelFirstD(keys=['image']),
#         transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
#         transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
#         transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
#     ])


# # === DATASET LOADING ===
# def get_loader(csv_path, cache_dir, test_fold, seed, batch_size, num_workers, external=False):
#     """
#     If external=True, ignore folds/splits and use the entire CSV as test set.
#     Otherwise, select rows where split == test_fold (numeric).
#     """
#     set_determinism(seed)
#     df = pd.read_csv(csv_path)

#     print("[data] rows:", len(df))

#     if external:
#         test_df = df.copy()
#         print(f"[data] external mode: selected {len(test_df)} rows (entire CSV)")
#     else:
#         if "split" not in df.columns:
#             raise ValueError("CSV must contain a 'split' column when not using --external.")
#         s = df["split"]
#         if pd.api.types.is_numeric_dtype(s) or s.astype(str).str.fullmatch(r"\d+").all():
#             mask = (s.astype(int) == int(test_fold))
#         else:
#             raise ValueError("Non-numeric 'split' values are not supported.")
#         test_df = df[mask].copy()
#         print(f"[data] fold={test_fold} seed={seed}: selected {len(test_df)} test rows")

#     if test_df.empty:
#         print("[warn] No test data selected.")
#         return None

#     testset = get_dataset_from_pd(test_df, get_transforms(), cache_dir)
#     print(f"[data] dataset size={len(testset)}; batch_size={batch_size}")
#     return DataLoader(
#         testset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=True,
#         persistent_workers=(num_workers > 0),
#     )


# # === LOSS ===
# loss_fn = L1Loss()


# def _tensor_info(t, name="tensor"):
#     try:
#         _min = t.min().item() if t.numel() > 0 else float("nan")
#         _max = t.max().item() if t.numel() > 0 else float("nan")
#     except Exception:
#         _min, _max = float("nan"), float("nan")
#     return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
#             f"numel={t.numel():,}, min={_min:.3g}, max={_max:.3g}")


# # === EVALUATION ===
# def evaluate_model(model, loader, ckpt_name="(unknown)"):
#     if loader is None:
#         return None
#     model.eval()
#     total_loss, count = 0.0, 0

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {ckpt_name}", leave=False), start=1):
#             images = batch["image"].to(DEVICE, non_blocking=True)

#             # Debug on first batch
#             if batch_idx == 1:
#                 print("[debug]", _tensor_info(images, "images"))
#                 try:
#                     print(f"[debug] const.INPUT_SHAPE_AE={getattr(const, 'INPUT_SHAPE_AE', None)} "
#                           f"const.RESOLUTION={getattr(const, 'RESOLUTION', None)}")
#                 except Exception:
#                     pass

#             # Guard: int32 indexing safety
#             if images.numel() > INT32_MAX:
#                 print(f"⚠️  [{ckpt_name}] images too large for int32 indexing: numel={images.numel():,}")
#                 return None

#             # Forward
#             try:
#                 recon, _, _ = model(images)
#             except Exception as e:
#                 print(f"⚠️  [{ckpt_name}] forward() failed on batch {batch_idx}: {type(e).__name__}: {e}")
#                 print("[debug]", _tensor_info(images, "images"))
#                 return None

#             # Debug on first output
#             if batch_idx == 1:
#                 print("[debug]", _tensor_info(recon, "recon"))

#             # Guards
#             if recon.numel() > INT32_MAX:
#                 print(f"⚠️  [{ckpt_name}] recon too large for int32 indexing: numel={recon.numel():,}")
#                 return None

#             if recon.shape != images.shape:
#                 print(f"⚠️  [{ckpt_name}] shape mismatch: recon {tuple(recon.shape)} vs images {tuple(images.shape)}")
#                 print("[hint] Check const.INPUT_SHAPE_AE/RESOLUTION and Spacing/Resize vs checkpoint arch.")
#                 return None

#             # Loss
#             try:
#                 loss = loss_fn(recon, images)
#             except RuntimeError as e:
#                 print(f"⚠️  [{ckpt_name}] loss failed: {e}")
#                 print("[debug]", _tensor_info(images, "images"))
#                 print("[debug]", _tensor_info(recon, "recon"))
#                 return None

#             total_loss += loss.item() * images.size(0)
#             count += images.size(0)

#     if count == 0:
#         print(f"[warn] No batches seen for {ckpt_name}.")
#         return None
#     return total_loss / count


# # === MAIN (supports single or multiple seeds) ===
# def main(args, seeds):
#     all_results = []  # list of tuples: (fold, seed, fname, loss)

#     fold_dirs = [d for d in sorted(os.listdir(args.ckpt_root_dir)) if re.match(r'fold_(\d+)', d)]
#     if not fold_dirs:
#         raise SystemExit(f"No fold_* directories found under {args.ckpt_root_dir}")

#     for fold_name in fold_dirs:
#         fold = int(re.match(r'fold_(\d+)', fold_name).group(1))
#         ckpt_dir = os.path.join(args.ckpt_root_dir, fold_name)
#         cache_dir = os.path.join(args.cache_dir, fold_name)  # reuse per-fold cache (works for external, too)

#         for seed in seeds:
#             print(f"\n🧪 Evaluating fold {fold} (seed={seed})"
#                   + (" on EXTERNAL CSV (ignoring split)" if args.external else f" on test split == {fold}") + "...")

#             test_loader = get_loader(args.csv_path, cache_dir, fold, seed,
#                                      args.batch_size, args.num_workers, external=args.external)
#             if test_loader is None:
#                 print(f"[warn] No test data for fold {fold}, seed {seed}; skipping.")
#                 continue

#             ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth") and "autoencoder" in f])
#             print(f"[eval] fold={fold}, seed={seed}: {len(ckpts)} checkpoints to evaluate")

#             seed_results = []
#             for i, fname in enumerate(ckpts, 1):
#                 ckpt_path = os.path.join(ckpt_dir, fname)
#                 print(f"[{i}/{len(ckpts)}] 📂 {fname}")
#                 try:
#                     model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
#                     val_loss = evaluate_model(model, test_loader, ckpt_name=fname)
#                     if val_loss is None:
#                         print("   ↳ skipped")
#                         continue
#                     print(f"   ↳ L1 = {val_loss:.6f}")
#                     seed_results.append((fold, seed, fname, val_loss))
#                 except Exception as e:
#                     print(f"⚠️ Failed on {fname}: {type(e).__name__}: {e}")

#             if seed_results:
#                 best_for_seed = min(seed_results, key=lambda x: x[3])
#                 print(f"✅ Best for fold {fold}, seed={seed}: {best_for_seed[2]} - L1 = {best_for_seed[3]:.6f}")
#                 all_results.extend(seed_results)
#             else:
#                 print(f"[warn] No results collected for fold={fold}, seed={seed}.")

#     # === Summaries ===
#     # 1) Best epoch per fold (mean across seeds if multiple seeds were provided)
#     best_per_fold = {}         # fold -> (best_epoch, mean_loss)
#     fold_epoch_losses = {}     # (fold, epoch) -> [losses across seeds]

#     for fold, seed, fname, loss in all_results:
#         m = re.search(r'epoch_(\d+)', fname)
#         if not m:
#             # If filenames don't have epoch info, we can't aggregate by epoch—skip
#             continue
#         epoch = int(m.group(1))
#         fold_epoch_losses.setdefault((fold, epoch), []).append(loss)

#     for (fold, epoch), losses in fold_epoch_losses.items():
#         mean_loss = sum(losses) / max(len(losses), 1)
#         if (fold not in best_per_fold) or (mean_loss < best_per_fold[fold][1]):
#             best_per_fold[fold] = (epoch, mean_loss)

#     # 2) Optional: write per-checkpoint results (fold, seed, filename, loss)
#     per_ckpt_csv = args.per_ckpt_csv
#     if per_ckpt_csv:
#         os.makedirs(os.path.dirname(per_ckpt_csv), exist_ok=True)
#         with open(per_ckpt_csv, "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(["fold", "seed", "checkpoint", "test_loss_L1"])
#             for fold, seed, fname, loss in sorted(all_results, key=lambda x: (x[0], x[1], x[2])):
#                 w.writerow([fold, seed, fname, f"{loss:.6f}"])
#         print(f"📄 Per-checkpoint results saved to: {per_ckpt_csv}")

#     # Write best-per-fold summary
#     os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
#     with open(args.output_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["fold", "best_epoch", "mean_test_loss_across_seeds"])
#         if not best_per_fold:
#             print("[warn] No best-per-fold results to write. Check data and epoch naming.")
#         for fold in sorted(best_per_fold):
#             epoch, loss = best_per_fold[fold]
#             writer.writerow([fold, epoch, f"{loss:.6f}"])
#     print(f"📄 Best-epoch-per-fold summary saved to: {args.output_csv}")


# # === CLI ===
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv_path", required=True, help="Path to dataset CSV file")
#     parser.add_argument("--cache_dir", required=True, help="Cache root (per-fold subdirs will be created)")
#     parser.add_argument("--ckpt_root_dir", required=True, help="Root checkpoint dir (with fold_* subfolders)")
#     parser.add_argument("--output_csv", default="test_losses_summary.csv", help="Where to save best-per-fold summary CSV")
#     parser.add_argument("--per_ckpt_csv", default="", help="(Optional) Save per-checkpoint results to this CSV")
#     parser.add_argument("--batch_size", type=int, default=2)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--external", action="store_true",
#                         help="Ignore 'split' and evaluate each fold's checkpoints on the full external CSV")

#     # Seed handling: either --seed OR --seeds
#     parser.add_argument("--seed", type=int, default=None, help="Single seed (optional, overrides --seeds)")
#     parser.add_argument("--seeds", default="0,1,2,3,4",
#                         help="Comma-separated list of seeds (used if --seed not given)")

#     args = parser.parse_args()

#     # Normalize seeds into a list
#     if args.seed is not None:
#         seeds_list = [args.seed]
#         print(f"🔹 Running with single seed: {args.seed}")
#     else:
#         seeds_list = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
#         print(f"🔹 Running with multiple seeds: {seeds_list}")

#     main(args, seeds_list)

