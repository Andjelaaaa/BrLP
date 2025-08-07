# import os
# import torch
# import warnings
# import pandas as pd
# from tqdm import tqdm
# from torch.nn import L1Loss
# from monai.utils import set_determinism
# from torch.utils.data import DataLoader

# from monai import transforms
# from brlp import const, get_dataset_from_pd, networks  # adjust import if needed

# set_determinism(0)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # === CONFIG ===
# csv_path     = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
# cache_dir    = "/home/andim/scratch/brlp/ae_cache"
# ckpt_dir     = "/home/andim/scratch/brlp/ae_output"
# batch_size   = 2
# num_workers  = 4

# # === MONAI TRANSFORMS (same as training) ===
# transforms_fn = transforms.Compose([
#     transforms.CopyItemsD(keys={'image_path'}, names=['image']),
#     transforms.LoadImageD(image_only=True, keys=['image']),
#     transforms.EnsureChannelFirstD(keys=['image']), 
#     transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
#     transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
#     transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
# ])

# # === DATA LOADER (validation) ===
# df = pd.read_csv(csv_path)
# valid_df = df[df.split == 'valid']
# validset = get_dataset_from_pd(valid_df, transforms_fn, cache_dir)
# valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers,
#                           shuffle=False, pin_memory=True, persistent_workers=True)

# # === LOSS FUNCTION ===
# loss_fn = L1Loss()

# # === EVALUATION FUNCTION ===
# def evaluate_model(model, loader):
#     model.eval()
#     total_loss = 0.0
#     count = 0
#     with torch.no_grad():
#         for batch in tqdm(loader, desc='Evaluating'):
#             images = batch["image"].to(DEVICE)
#             recon, _, _ = model(images)
#             loss = loss_fn(recon, images)
#             total_loss += loss.item() * images.size(0)
#             count += images.size(0)
#     return total_loss / count

# # === EVALUATION LOOP ===
# results = []

# for fname in sorted(os.listdir(ckpt_dir)):
#     if fname.endswith('.pth') and 'autoencoder' in fname:
#         ckpt_path = os.path.join(ckpt_dir, fname)
#         print(f'\n📂 Loading {fname}')
#         model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
#         val_loss = evaluate_model(model, valid_loader)
#         print(f'{fname}: val L1 loss = {val_loss:.6f}')
#         results.append((fname, val_loss))

# # === SAVE RESULTS TO CSV ===
# csv_output_path = os.path.join(ckpt_dir, "validation_losses.csv")
# with open(csv_output_path, mode="w", newline="") as f:
#     import csv
#     writer = csv.writer(f)
#     writer.writerow(["checkpoint", "val_loss"])
#     for fname, val_loss in sorted(results, key=lambda x: x[1]):
#         writer.writerow([fname, val_loss])

# print(f"\n✅ Validation results saved to: {csv_output_path}")

# # === TOP MODELS (OPTIONAL PRINT) ===
# print("\n🔍 Top 5 models:")
# for fname, val_loss in sorted(results, key=lambda x: x[1])[:5]:
#     print(f'{fname}: {val_loss:.6f}')

###########################################################################

# import os
# import csv
# import torch
# import warnings
# import pandas as pd
# from tqdm import tqdm
# from torch.nn import L1Loss
# from monai.utils import set_determinism
# from torch.utils.data import DataLoader

# from monai import transforms
# from brlp import const, get_dataset_from_pd, networks  # adjust import if needed

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # === CONFIG ===
# csv_path     = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
# cache_dir    = "/home/andim/scratch/brlp/ae_cache_long"
# ckpt_dir     = "/home/andim/scratch/brlp/ae_output_long"
# batch_size   = 2
# num_workers  = 4
# seeds        = [0, 1, 2, 3, 4]  # seeds to evaluate

# # === MONAI TRANSFORMS (same as training) ===
# def get_transforms():
#     return transforms.Compose([
#         transforms.CopyItemsD(keys={'image_path'}, names=['image']),
#         transforms.LoadImageD(image_only=True, keys=['image']),
#         transforms.EnsureChannelFirstD(keys=['image']), 
#         transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
#         transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
#         transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
#     ])

# # === DATASET LOADING ===
# def get_valid_loader(seed):
#     set_determinism(seed)
#     df = pd.read_csv(csv_path)
#     valid_df = df[df.split == 'valid']
#     validset = get_dataset_from_pd(valid_df, get_transforms(), cache_dir)
#     return DataLoader(validset, batch_size=batch_size, num_workers=num_workers,
#                       shuffle=False, pin_memory=True, persistent_workers=True)

# # === LOSS FUNCTION ===
# loss_fn = L1Loss()

# # === EVALUATION FUNCTION ===
# def evaluate_model(model, loader):
#     model.eval()
#     total_loss = 0.0
#     count = 0
#     with torch.no_grad():
#         for batch in tqdm(loader, desc='Evaluating', leave=False):
#             images = batch["image"].to(DEVICE)
#             recon, _, _ = model(images)
#             loss = loss_fn(recon, images)
#             total_loss += loss.item() * images.size(0)
#             count += images.size(0)
#     return total_loss / count

# # === MAIN EVALUATION LOOP ===
# all_results = []

# for seed in seeds:
#     print(f"\n🌱 Evaluating with seed {seed}...")
#     valid_loader = get_valid_loader(seed)
#     seed_results = []

#     for fname in sorted(os.listdir(ckpt_dir)):
#         if fname.endswith('.pth') and 'autoencoder' in fname:
#             ckpt_path = os.path.join(ckpt_dir, fname)
#             print(f'📂 Loading {fname}')
#             model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
#             val_loss = evaluate_model(model, valid_loader)
#             print(f'{fname}: val L1 loss = {val_loss:.6f}')
#             seed_results.append((seed, fname, val_loss))
    
#     all_results.extend(seed_results)

#     # Show Top 5 models for this seed
#     top5 = sorted(seed_results, key=lambda x: x[2])[:5]
#     print(f"\n🔍 Top 5 models for seed {seed}:")
#     for _, fname, val_loss in top5:
#         print(f'{fname}: {val_loss:.6f}')

# # === SAVE ALL RESULTS TO CSV ===
# csv_output_path = os.path.join("validation_losses_over_seeds_output_long.csv")
# with open(csv_output_path, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["seed", "checkpoint", "val_loss"])
#     for seed, fname, val_loss in sorted(all_results, key=lambda x: (x[0], x[2])):
#         writer.writerow([seed, fname, val_loss])

# print(f"\n✅ Validation results across seeds saved to: {csv_output_path}")

##########################################################################################
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
    test_df = df[df.split == {test_fold}]
    testset = get_dataset_from_pd(test_df, get_transforms(), cache_dir)
    return DataLoader(testset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=True, persistent_workers=True)

# === LOSS FUNCTION ===
loss_fn = L1Loss()

# === MODEL EVALUATION ===
def evaluate_model(model, loader):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            images = batch["image"].to(DEVICE)
            recon, _, _ = model(images)
            loss = loss_fn(recon, images)
            total_loss += loss.item() * images.size(0)
            count += images.size(0)
    return total_loss / count

# === MAIN ===
def main(args):
    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for fold_name in sorted(os.listdir(args.ckpt_root_dir)):
        match = re.match(r'fold(\d+)', fold_name)
        if not match:
            continue
        fold = int(match.group(1))
        ckpt_dir = os.path.join(args.ckpt_root_dir, fold_name)
        cache_dir = os.path.join(args.cache_dir, fold_name)

        print(f"\n🧪 Evaluating fold {fold} using test set test{fold}...")

        for seed in seeds:
            print(f"\n🌱 Seed {seed}")
            test_loader = get_loader(args.csv_path, cache_dir, fold, seed,
                                     args.batch_size, args.num_workers)
            seed_results = []

            for fname in sorted(os.listdir(ckpt_dir)):
                if fname.endswith('.pth') and 'autoencoder' in fname:
                    ckpt_path = os.path.join(ckpt_dir, fname)
                    print(f'📂 Loading {fname}')
                    try:
                        model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
                        val_loss = evaluate_model(model, test_loader)
                        print(f'{fname}: test L1 loss = {val_loss:.6f}')
                        seed_results.append((fold, seed, fname, val_loss))
                    except Exception as e:
                        print(f"⚠️ Failed to load {fname}: {e}")

            # Save top-1 for seed/fold
            if seed_results:
                best_model = sorted(seed_results, key=lambda x: x[3])[0]
                print(f"\n✅ Best model for fold {fold}, seed {seed}: {best_model[2]} - L1 = {best_model[3]:.6f}")
                all_results.append(best_model)

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
