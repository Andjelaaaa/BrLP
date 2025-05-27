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
image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-10007_ses-002_brain.nii.gz'
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

# === Load image and transform ===
input_dict = {"image_path": image_path}
image_tensor = transforms_fn(input_dict)["image"].unsqueeze(0).to(DEVICE)  # shape: (1, 1, H, W, D)
print("Original image min/max:", image_tensor.min().item(), image_tensor.max().item())
# === Convert to NumPy for patch insertion ===
image_np = image_tensor.cpu().numpy().copy()  # shape: (1, 1, H, W, D)
modified_np = image_np.copy()
print("Modified image min/max:", modified_np.min().item(), modified_np.max().item())

# === Get shape ===
_, _, H, W, D = image_tensor.shape

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

# === Save modified volume to NIfTI ===
# Load affine and header from original image
orig_img_nii = nib.load(image_path)
affine = orig_img_nii.affine
header = orig_img_nii.header

# Save modified version
modified_nifti = nib.Nifti1Image(modified_np.squeeze(), affine, header)
nib.save(modified_nifti, output_nifti_path)

print(f"✅ Modified image with patch saved to: {output_nifti_path}")

