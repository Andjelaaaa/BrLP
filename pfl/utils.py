import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

# === Load data ===
def load_data(latent_trajectories_csv, metadata_csv, latent_shape=(3, 14, 18, 14)):
    latent_dim = np.prod(latent_shape)
    df_latents = pd.read_csv(latent_trajectories_csv)
    df_metadata = pd.read_csv(metadata_csv)
    latent_cols = [col for col in df_latents.columns if col.startswith("latent_")]

    paired_data = defaultdict(list)
    for _, row in df_latents.iterrows():
        paired_data[row["subject_id"]].append(row)

    start_latents = []
    delta_vectors = []
    age_diffs = []
    subject_ids = []
    image_uids = []
    splits = []

    for subject_id, scans in paired_data.items():
        if len(scans) < 2:
            continue
        scans = sorted(scans, key=lambda r: r["age"])

        for i, j in combinations(range(len(scans)), 2):
            row_start = scans[i]
            row_follow = scans[j]
            age_diff = row_follow["age"] * 6 - row_start["age"] * 6
            if age_diff <= 0:
                continue

            start_latent = row_start[latent_cols].to_numpy()
            follow_latent = row_follow[latent_cols].to_numpy()
            delta_vector = (follow_latent - start_latent) / age_diff

            start_latents.append(start_latent)
            delta_vectors.append(delta_vector)
            age_diffs.append(age_diff)
            subject_ids.append(row_follow["subject_id"])
            image_uids.append(row_follow["image_uid"])
            splits.append(row_follow["split"])

    return (
        np.vstack(start_latents).astype(np.float32),
        np.vstack(delta_vectors).astype(np.float32),
        np.array(age_diffs, dtype=np.float32),
        subject_ids, image_uids,
        df_latents, df_metadata, latent_cols,
        np.array(splits)
    )

def log_recon_comparison_to_wandb(
    sample,
    starting_recon_np,
    real_recon_np,
    predicted_recon_np,
    step=None,
    tag="Reconstruction Comparison"
):
    """
    Logs a 3x3 grid of reconstruction slices to wandb.

    Args:
        sample (dict): Sample metadata.
        starting_recon_np (np.ndarray): Starting reconstruction [1, D, H, W].
        real_recon_np (np.ndarray): True follow-up reconstruction [1, D, H, W].
        predicted_recon_np (np.ndarray): Predicted reconstruction [1, D, H, W].
        step (int, optional): wandb step to log under.
        tag (str): wandb log tag.
    """
    mid_slice = predicted_recon_np.shape[-1] // 2
    cor_slice = predicted_recon_np.shape[2] // 2
    sag_slice = predicted_recon_np.shape[1] // 2

    # Axial slices
    starting_slice_ax = starting_recon_np[0, :, :, mid_slice]
    real_slice_ax = real_recon_np[0, :, :, mid_slice]
    predicted_slice_ax = predicted_recon_np[0, :, :, mid_slice]

    # Coronal slices
    starting_slice_cor = starting_recon_np[0, :, cor_slice, :]
    real_slice_cor = real_recon_np[0, :, cor_slice, :]
    predicted_slice_cor = predicted_recon_np[0, :, cor_slice, :]

    # Sagittal slices
    starting_slice_sag = starting_recon_np[0, sag_slice, :, :]
    real_slice_sag = real_recon_np[0, sag_slice, :, :]
    predicted_slice_sag = predicted_recon_np[0, sag_slice, :, :]

    titles = [
        f"Start Axial | Age: {sample['start_age']:.2f}",
        f"Real Axial | Age: {sample['followup_age']:.2f}",
        "Pred Axial",
        "Start Coronal", "Real Coronal", "Pred Coronal",
        "Start Sagittal", "Real Sagittal", "Pred Sagittal"
    ]

    images = [
        starting_slice_ax, real_slice_ax, predicted_slice_ax,
        starting_slice_cor, real_slice_cor, predicted_slice_cor,
        starting_slice_sag, real_slice_sag, predicted_slice_sag
    ]

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{sample['subject_id']} | ΔAge = {sample['followup_age'] - sample['start_age']:.2f} yrs "
        f"| L2 = {sample['l2_error']:.2f} | MSE = {sample['mse']:.2f} | ΔErr = {sample['error_diff']:.2f}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    tmp_path = f"{sample['subject_id']}_{sample['image_uid']}_wandb_recon_tmp.png"
    plt.savefig(tmp_path, dpi=120)
    plt.close()

    # Upload to wandb
    wandb.log({tag: wandb.Image(tmp_path)}, step=step)
    os.remove(tmp_path)