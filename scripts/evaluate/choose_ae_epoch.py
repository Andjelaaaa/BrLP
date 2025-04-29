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


import os
import csv
import torch
import warnings
import pandas as pd
from tqdm import tqdm
from torch.nn import L1Loss
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from monai import transforms
from brlp import const, get_dataset_from_pd, networks  # adjust import if needed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === CONFIG ===
csv_path     = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
cache_dir    = "/home/andim/scratch/brlp/ae_cache"
ckpt_dir     = "/home/andim/scratch/brlp/ae_output"
batch_size   = 2
num_workers  = 4
seeds        = [0, 1, 2, 3, 4]  # seeds to evaluate

# === MONAI TRANSFORMS (same as training) ===
def get_transforms():
    return transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

# === DATASET LOADING ===
def get_valid_loader(seed):
    set_determinism(seed)
    df = pd.read_csv(csv_path)
    valid_df = df[df.split == 'valid']
    validset = get_dataset_from_pd(valid_df, get_transforms(), cache_dir)
    return DataLoader(validset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=True, persistent_workers=True)

# === LOSS FUNCTION ===
loss_fn = L1Loss()

# === EVALUATION FUNCTION ===
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

# === MAIN EVALUATION LOOP ===
all_results = []

for seed in seeds:
    print(f"\n🌱 Evaluating with seed {seed}...")
    valid_loader = get_valid_loader(seed)
    seed_results = []

    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith('.pth') and 'autoencoder' in fname:
            ckpt_path = os.path.join(ckpt_dir, fname)
            print(f'📂 Loading {fname}')
            model = networks.init_autoencoder(ckpt_path).to(DEVICE).eval()
            val_loss = evaluate_model(model, valid_loader)
            print(f'{fname}: val L1 loss = {val_loss:.6f}')
            seed_results.append((seed, fname, val_loss))
    
    all_results.extend(seed_results)

    # Show Top 5 models for this seed
    top5 = sorted(seed_results, key=lambda x: x[2])[:5]
    print(f"\n🔍 Top 5 models for seed {seed}:")
    for _, fname, val_loss in top5:
        print(f'{fname}: {val_loss:.6f}')

# === SAVE ALL RESULTS TO CSV ===
csv_output_path = os.path.join(ckpt_dir, "validation_losses_over_seeds.csv")
with open(csv_output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "checkpoint", "val_loss"])
    for seed, fname, val_loss in sorted(all_results, key=lambda x: (x[0], x[2])):
        writer.writerow([seed, fname, val_loss])

print(f"\n✅ Validation results across seeds saved to: {csv_output_path}")
