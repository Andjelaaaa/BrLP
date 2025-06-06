import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import subprocess
import nibabel as nib


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
    start_ages = []
    followup_ages = []
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
            age_start = row_start["age"] * 6
            age_follow = row_follow["age"] * 6
            age_diff = age_follow - age_start
            if age_diff <= 0:
                continue

            start_latent = row_start[latent_cols].to_numpy()
            follow_latent = row_follow[latent_cols].to_numpy()
            delta_vector = (follow_latent - start_latent) / age_diff

            start_latents.append(start_latent)
            delta_vectors.append(delta_vector)
            age_diffs.append(age_diff)
            start_ages.append(age_start)
            followup_ages.append(age_follow)
            subject_ids.append(row_follow["subject_id"])
            image_uids.append(row_follow["image_uid"])
            splits.append(row_follow["split"])

    return (
        np.vstack(start_latents).astype(np.float32),
        np.vstack(delta_vectors).astype(np.float32),
        np.array(age_diffs, dtype=np.float32),
        np.array(start_ages, dtype=np.float32),
        np.array(followup_ages, dtype=np.float32),
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

    print('The shown slices are...:')
    print(mid_slice, cor_slice, sag_slice)
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
        f"Start T\u2081 Axial | Age: {sample['start_age']:.2f}",
        f"Real T\u2082 Axial | Age: {sample['followup_age']:.2f}",
        "Pred T\u2082 Axial",
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

def register_and_jacobian_cli(pred_path, true_path, output_dir=".", tag="recon", visualize=True):
    """
    Use ANTs command-line tools to register predicted image to true image and compute Jacobian determinant.

    Args:
        pred_path (str): Path to predicted reconstruction image (.nii.gz).
        true_path (str): Path to true reconstruction image (.nii.gz).
        output_dir (str): Directory to save outputs.
        tag (str): Prefix for saved files.
        visualize (bool): Whether to plot the Jacobian slice.

    Returns:
        jac_path (str): Path to saved Jacobian image.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Run antsRegistration
    out_prefix = os.path.join(output_dir, f"{tag}_")
    reg_cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{out_prefix},{out_prefix}warped.nii.gz]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{true_path},{pred_path},1]",
        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{true_path},{pred_path},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox"
    ]
    print('REGISTRATION COMMAND')
    print(reg_cmd)
    print("🔄 Running registration...")
    subprocess.run(reg_cmd, check=True)

    # Step 2: Compute Jacobian from warp
    warp_field = f"{out_prefix}1Warp.nii.gz"
    jac_path = f"{out_prefix}jacobian.nii.gz"
    jac_cmd = [
        "CreateJacobianDeterminantImage", "3",
        warp_field,
        jac_path,
        "1",  # 1 = use log (0 = no log)
        "0"   # 0 = don't invert sign
    ]
    print("🧮 Creating Jacobian determinant...")
    subprocess.run(jac_cmd, check=True)

    # Step 3: Optional visualization
    if visualize:
        print("📈 Visualizing Jacobian slice...")
        img = nib.load(jac_path)
        data = img.get_fdata()
        mid_slice = data.shape[2] // 2
        plt.imshow(np.rot90(data[:, :, mid_slice]), cmap="coolwarm", vmin=0.5, vmax=1.5)
        plt.title("Jacobian Determinant (Axial Slice)")
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return jac_path