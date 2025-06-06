import os
import numpy as np
import torch
import random
import nibabel as nib
import matplotlib.pyplot as plt
from monai import transforms
from brlp import init_autoencoder, const

# === CONFIGURATION ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"
# image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-10007_ses-002_brain.nii.gz'
# image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-051407_ses-12mo_brain.nii.gz'
# mask_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-051407_ses-12mo_mask.nii.gz'
# image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-469090_ses-12mo_brain.nii.gz'
# mask_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-469090_ses-12mo_mask.nii.gz'
image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-011228_ses-12mo_brain.nii.gz'
mask_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-011228_ses-12mo_mask.nii.gz'
output_nifti_path = image_path.replace(".nii", "_with_patch.nii")

# === Load model ===
autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

# === MONAI transforms for preprocessing ===
transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image']),
    transforms.EnsureChannelFirstD(keys=['image']),
    transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
])

mask_transforms = transforms.Compose([
    transforms.LoadImageD(keys=['mask'], image_only=True),
    transforms.EnsureChannelFirstD(keys=['mask']),
    transforms.SpacingD(
        pixdim=const.RESOLUTION,
        keys=['mask'],
        mode='nearest'
    ),
    transforms.ResizeWithPadOrCropD(
        keys=['mask'],
        spatial_size=const.INPUT_SHAPE_AE,
        mode='constant'   # pad with 0 where needed
    ),
    # no ScaleIntensityD here
])

# ─── load & preprocess mask ──────────────────────────────────────
mask_input  = {'mask': mask_path}
mask_tensor = mask_transforms(mask_input)['mask']   # shape: (1, H, W, D)
mask_np     = mask_tensor.cpu().numpy().squeeze()    # shape: (H, W, D)


# === Load image and transform ===
input_dict = {"image_path": image_path}
image_tensor = transforms_fn(input_dict)["image"].unsqueeze(0).to(DEVICE)  # shape: (1, 1, H, W, D)
print("Original image min/max:", image_tensor.min().item(), image_tensor.max().item())
# === Convert to NumPy for patch insertion ===
image_np = image_tensor.cpu().numpy().copy()  # shape: (1, 1, H, W, D)
modified_np = image_np.copy()
print("Modified image min/max:", modified_np.min().item(), modified_np.max().item())

apply_patch = False 

# copy the original into modified
modified_np = image_np.copy()

# === Get shape ===
_, _, H, W, D = image_tensor.shape

if apply_patch:
    # === Random cube patch ===
    square_size = random.randint(1, 4)
    x = random.randint(0, H - square_size)
    y = random.randint(0, W - square_size)
    z = random.randint(0, D - square_size)

    # Insert a bright cube
    patch_intensity = 1.0  
    modified_np[0, 0, x:x+square_size, y:y+square_size, z:z+square_size] = patch_intensity

    # === Convert back to tensor for AE processing ===
    modified_tensor = torch.from_numpy(modified_np).float().to(DEVICE)

    # === Encode and decode ===
    with torch.no_grad():
        recon_orig = autoencoder.decode(autoencoder.encode(image_tensor)[0])
        recon_mod  = autoencoder.decode(autoencoder.encode(modified_tensor)[0])

    # === Extract middle slice of patch for visualization ===
    patch_center_z = z + square_size // 2

    orig_slice = image_np[0, 0, :, :, patch_center_z]
    mod_slice  = modified_np[0, 0, :, :, patch_center_z]
    diff_slice = np.abs(mod_slice - orig_slice)

    # === Plot the result ===
    # === Also extract reconstructions for the same slice ===
    recon_orig_np = recon_orig.cpu().numpy().squeeze()
    recon_mod_np  = recon_mod.cpu().numpy().squeeze()
    recon_diff_np = np.abs(recon_mod_np - recon_orig_np)

    # === Slice all at patch_center_z ===
    recon_orig_slice = recon_orig_np[:, :, patch_center_z]
    recon_mod_slice  = recon_mod_np[:, :, patch_center_z]
    recon_diff_slice = recon_diff_np[:, :, patch_center_z]

    # === Create 2x3 plot ===
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title("Original Input (slice {})".format(patch_center_z))
    plt.imshow(orig_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Modified Input with Patch")
    plt.imshow(mod_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Input Difference (abs)")
    plt.imshow(diff_slice, cmap='hot')
    plt.colorbar()
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Reconstruction: Original")
    plt.imshow(recon_orig_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Reconstruction: Modified")
    plt.imshow(recon_mod_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Reconstruction Diff (abs)")
    plt.imshow(recon_diff_slice, cmap='hot')
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('varying_squares_ae_recon.png')
    plt.show()

else:
    # no patch → just reconstruct the original
    print("⚠️ No patch applied; reconstructing the raw input.")
    # pick a slice for visualization (e.g. the middle)
    patch_center_z = D // 2

    # extract original slice
    orig_slice = image_np[0, 0, :, :, patch_center_z]

    # run autoencoder on the clean input
    with torch.no_grad():
        recon = autoencoder.decode(autoencoder.encode(image_tensor)[0])

    recon_np    = recon.cpu().numpy().squeeze()
    recon_slice = recon_np[:, :, patch_center_z]

    # difference between recon and original
    diff_slice  = np.abs(recon_slice - orig_slice)

    # 1) pick the same slice
    mask_slice = mask_np[:, :, patch_center_z]      # shape (H, W)

    # 2) compute masked diffs
    brain_diffs = diff_slice[mask_slice == 1]       # only where mask==1

    # 3) mean absolute error inside brain
    mean_brain_diff = float(brain_diffs.mean())     if brain_diffs.size else 0.0

    # plot only three panels: orig, recon, diff
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Original (slice {patch_center_z})")
    plt.imshow(orig_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    plt.imshow(recon_slice, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Abs Diff (brain mask): mean={mean_brain_diff:.3f}")
    plt.imshow(diff_slice, cmap='hot')
    plt.colorbar(shrink=0.6)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('recon_only_diff.png')



# # === Save modified volume to NIfTI ===
# # Load affine and header from original image
# orig_img_nii = nib.load(image_path)
# affine = orig_img_nii.affine
# header = orig_img_nii.header

# # Save modified version
# modified_nifti = nib.Nifti1Image(modified_np.squeeze(), affine, header)
# nib.save(modified_nifti, output_nifti_path)

# print(f"✅ Modified image with patch saved to: {output_nifti_path}")

