import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from monai import transforms
from brlp import init_autoencoder, const
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === CONFIG ===
aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"
latent_mean_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy"
latent_std_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"
image_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-1098_brain.nii.gz'
output_html_path = "full_deviation_map_interactive.html"

# === Load model and stats ===
autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()
latent_mean = np.load(latent_mean_path)
latent_std = np.load(latent_std_path)

# === Define MONAI transforms ===
transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image']),
    transforms.EnsureChannelFirstD(keys=['image']),
    transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
])

# === Preprocess input image ===
input_dict = {"image_path": image_path}
image_tensor = transforms_fn(input_dict)["image"].unsqueeze(0).to(DEVICE)

# === Encode input ===
with torch.no_grad():
    z_mu, _ = autoencoder.encode(image_tensor)
    z_mu_np = z_mu.squeeze(0).cpu().numpy()

z_flat = z_mu_np.flatten()

# === Z-Score the latent ===
z_score = (z_flat - latent_mean) / latent_std

# === Prepare different threshold reconstructions ===
thresholds = np.linspace(0, 5, num=21)
original_image = image_tensor.squeeze(0).cpu().numpy()

projected_images = []
residual_maps = []

for thresh in thresholds:
    z_score_thresh = np.clip(z_score, -thresh, thresh)
    z_proj = z_score_thresh * latent_std + latent_mean
    z_proj_tensor = torch.tensor(z_proj.reshape(z_mu_np.shape), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon = autoencoder.decode(z_proj_tensor)

    recon_np = recon.squeeze(0).cpu().numpy()
    residual = np.abs(original_image - recon_np)

    projected_images.append(recon_np)
    residual_maps.append(residual)

# # === Prepare slice indices ===
# mid_axial = original_image.shape[-1] // 2
# mid_sagittal = original_image.shape[2] // 2
# mid_coronal = original_image.shape[1] // 2

# # === Create subplots ===
# fig = make_subplots(
#     rows=3, cols=3,
#     subplot_titles=[
#         "Axial Input", "Axial Healthy", "Axial Deviation",
#         "Coronal Input", "Coronal Healthy", "Coronal Deviation",
#         "Sagittal Input", "Sagittal Healthy", "Sagittal Deviation"
#     ],
#     horizontal_spacing=0.02,
#     vertical_spacing=0.02
# )

# # === Helper: add a heatmap at specific location ===
# def add_heatmaps(i, row, col, view_axis, slice_idx):
#     if view_axis == 2:
#         img = original_image[0, :, :, slice_idx]
#         proj = projected_images[i][0, :, :, slice_idx]
#         resid = residual_maps[i][0, :, :, slice_idx]
#     elif view_axis == 1:
#         img = original_image[0, :, slice_idx, :]
#         proj = projected_images[i][0, :, slice_idx, :]
#         resid = residual_maps[i][0, :, slice_idx, :]
#     else:
#         img = original_image[0, slice_idx, :, :]
#         proj = projected_images[i][0, slice_idx, :, :]
#         resid = residual_maps[i][0, slice_idx, :, :]

#     return [
#         go.Heatmap(z=img, colorscale="gray", showscale=False),
#         go.Heatmap(z=proj, colorscale="gray", showscale=False),
#         go.Heatmap(z=resid, colorscale="hot", colorbar=dict(title="Deviation") if (row == 1 and col == 3) else None)
#     ]

# # === Add initial traces (for first threshold) ===
# initial_traces = []
# for view_idx, (row, view_axis, slice_idx) in enumerate([(1,2,mid_axial), (2,1,mid_coronal), (3,0,mid_sagittal)]):
#     heatmaps = add_heatmaps(0, row, 1, view_axis, slice_idx)
#     fig.add_trace(heatmaps[0], row=row, col=1)
#     fig.add_trace(heatmaps[1], row=row, col=2)
#     fig.add_trace(heatmaps[2], row=row, col=3)

# # === Create frames for slider animation ===
# frames = []
# for frame_idx, thresh in enumerate(thresholds):
#     frame_data = []
#     for view_idx, (row, view_axis, slice_idx) in enumerate([(1,2,mid_axial), (2,1,mid_coronal), (3,0,mid_sagittal)]):
#         hmaps = add_heatmaps(frame_idx, row, 1, view_axis, slice_idx)
#         frame_data.extend(hmaps)

#     frames.append(go.Frame(data=frame_data, name=f"Threshold {thresh:.2f}σ"))

# fig.frames = frames

# # === Add slider ===
# sliders = [{
#     "steps": [
#         {
#             "args": [[f"Threshold {thresh:.2f}σ"], {"frame": {"duration": 0, "redraw": True},
#                                                      "mode": "immediate", "transition": {"duration": 0}}],
#             "label": f"{thresh:.2f}σ",
#             "method": "animate",
#         }
#         for thresh in thresholds
#     ],
#     "currentvalue": {"prefix": "Z-Threshold: "}
# }]

# # === Final layout ===
# fig.update_layout(
#     width=1800,
#     height=1500,
#     title="Deviation Map Viewer (Input / Healthy / Deviation Heatmaps)",
#     sliders=sliders,
#     updatemenus=[{
#         "type": "buttons",
#         "buttons": [{"label": "Play", "method": "animate", "args": [None]}]
#     }]
# )

# # === Save ===
# fig.write_html(output_html_path)
# print(f"✅ Saved interactive deviation viewer to: {output_html_path}")

# === Load and transform brain mask ===
# Load original binary brain mask
mask_path = '/home/andim/projects/def-bedelb/andim/brlp-data/sub-1098_mask.nii.gz'
raw_mask = nib.load(mask_path).get_fdata()

# Define transforms for binary mask
mask_transforms = transforms.Compose([
    transforms.CopyItemsD(keys={'mask_path'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image']),
    transforms.EnsureChannelFirstD(keys=["image"]),
    transforms.SpacingD(keys=["image"], pixdim=const.RESOLUTION, mode='nearest'),
    transforms.ResizeWithPadOrCropD(keys=["image"], spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
])

# Apply transforms to the mask directly
input_dict = {"mask_path": mask_path}
mask_tensor = mask_transforms(input_dict)["image"] #.unsqueeze(0).to(DEVICE)
brain_mask = (mask_tensor.squeeze(0) > 0.5).astype(np.float32)

# === Extract subject ID ===
subject_id = os.path.basename(mask_path).split('_')[0]

# === Prepare residual metrics ===
mean_abs_residuals = []

for i in range(len(thresholds)):
    residual = residual_maps[i][0]
    masked_residual = residual * brain_mask
    mean_abs_residual = masked_residual.sum() / brain_mask.sum()
    mean_abs_residuals.append(mean_abs_residual)

# === Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(thresholds, mean_abs_residuals, marker='o')
plt.xlabel("Z-Score Threshold (σ)")
plt.ylabel("Mean Absolute Residual (inside brain)")
plt.title(f"Deviation vs Threshold (Brain Only) for {subject_id}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"threshold_vs_deviation_{subject_id}.png")
plt.show()

print(f"✅ Saved plot: threshold_vs_deviation_{subject_id}.png")


