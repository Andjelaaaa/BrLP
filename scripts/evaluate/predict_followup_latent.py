# import os
# import numpy as np
# import pandas as pd
# import torch
# import nibabel as nib
# from scipy.interpolate import RBFInterpolator
# from sklearn.decomposition import PCA
# from itertools import combinations
# from collections import defaultdict
# from monai import transforms
# from brlp import init_autoencoder, const
# import matplotlib.pyplot as plt
# from monai import transforms

# transforms_fn = transforms.Compose([
#     transforms.CopyItemsD(keys={'image_path'}, names=['image']),
#     transforms.LoadImageD(image_only=True, keys=['image']),
#     transforms.EnsureChannelFirstD(keys=['image']),
#     transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
#     transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
#     transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
# ])


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ################################################################################
# # CONFIG
# ################################################################################
# latent_shape = (3, 14, 18, 14)
# latent_dim = np.prod(latent_shape)

# latent_trajectories_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
# metadata_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
# aekl_checkpoint = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"
# latent_mean_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy"
# latent_std_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"

# # How many dimensions to keep for PCA
# PCA_DIM = 128
# RBF_SMOOTHING = 1e-2

# ################################################################################
# # 1. Load data
# ################################################################################
# df_latents = pd.read_csv(latent_trajectories_csv)
# df_metadata = pd.read_csv(metadata_csv)

# latent_cols = [col for col in df_latents.columns if col.startswith("latent_")]

# ################################################################################
# # 2. Build Training Set for PCA and RBF
# ################################################################################
# train_df = df_latents[df_latents["split"] == "train"]

# paired_train = defaultdict(list)
# for _, row in train_df.iterrows():
#     paired_train[row["subject_id"]].append(row)

# start_latents = []
# delta_vectors = []

# for subject_id, scans in paired_train.items():
#     if len(scans) < 2:
#         continue
#     scans = sorted(scans, key=lambda r: r["age"])
    
#     for i, j in combinations(range(len(scans)), 2):
#         row_start = scans[i]
#         row_follow = scans[j]

#         age_diff = row_follow["age"] * 6 - row_start["age"] * 6
#         if age_diff <= 0:
#             continue

#         start_latent = row_start[latent_cols].to_numpy()
#         follow_latent = row_follow[latent_cols].to_numpy()
#         delta_vector = (follow_latent - start_latent) / age_diff

#         start_latents.append(start_latent)
#         delta_vectors.append(delta_vector)

# start_latents = np.vstack(start_latents)
# delta_vectors = np.vstack(delta_vectors)

# print(f"Training samples: {start_latents.shape[0]}, Latent dimension: {start_latents.shape[1]}")

# ################################################################################
# # 3. Train PCA
# ################################################################################
# print("Training PCA...")
# pca = PCA(n_components=PCA_DIM)
# start_latents_pca = pca.fit_transform(start_latents)
# delta_vectors_pca = pca.transform(delta_vectors)

# print(f"PCA transformed shape: {start_latents_pca.shape}")

# ################################################################################
# # 4. Train RBF Interpolator
# ################################################################################
# print("Training RBF Interpolator...")
# rbf = RBFInterpolator(start_latents_pca, delta_vectors_pca, smoothing=RBF_SMOOTHING)

# ################################################################################
# # 5. Predict on Test Set
# ################################################################################
# test_df = df_latents[df_latents["split"] == "test"]

# paired_test = defaultdict(list)
# for _, row in test_df.iterrows():
#     paired_test[row["subject_id"]].append(row)

# errors = []

# for subject_id, scans in paired_test.items():
#     if len(scans) < 2:
#         continue
#     scans = sorted(scans, key=lambda r: r["age"])
    
#     for i, j in combinations(range(len(scans)), 2):
#         row_start = scans[i]
#         row_follow = scans[j]
#         delta_t = row_follow["age"] * 6 - row_start["age"] * 6
#         if delta_t <= 0:
#             continue

#         start_latent = row_start[latent_cols].to_numpy()
#         true_follow_latent = row_follow[latent_cols].to_numpy()

#         # Predict
#         start_latent_pca = pca.transform(start_latent[np.newaxis, :])
#         delta_pred_pca = rbf(start_latent_pca)[0]
#         predicted_pca = start_latent_pca + delta_pred_pca * delta_t
#         predicted_flat_latent = pca.inverse_transform(predicted_pca)

#         l2 = np.linalg.norm(predicted_flat_latent - true_follow_latent)
#         mse = np.mean((predicted_flat_latent - true_follow_latent) ** 2)

#         errors.append({
#             "subject_id": subject_id,
#             "image_uid": row_follow["image_uid"],
#             "start_age": row_start["age"] * 6,
#             "followup_age": row_follow["age"] * 6,
#             "l2_error": l2,
#             "mse": mse,
#             "predicted_latent": predicted_flat_latent,
#             "true_followup_latent": true_follow_latent
#         })

# ################################################################################
# # 6. Pick the Best Sample (Lowest L2 Error)
# ################################################################################
# best_sample = min(errors, key=lambda x: x["l2_error"])

# print(f"\n✅ Best sample: {best_sample['subject_id']} | {best_sample['image_uid']} | L2 error: {best_sample['l2_error']:.4f}")

# ################################################################################
# # 7. Decode Predicted Latent to Image
# ################################################################################

# # Load trained autoencoder
# autoencoder = init_autoencoder(aekl_checkpoint).to(DEVICE).eval()

# # Load latent normalization stats
# latent_mean = np.load(latent_mean_path)
# latent_std = np.load(latent_std_path)

# # Undo normalization if needed (depends on how your AE expects inputs)

# predicted_latent_reshaped = best_sample["predicted_latent"].reshape(latent_shape)
# z_proj_tensor = torch.tensor(predicted_latent_reshaped, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# with torch.no_grad():
#     predicted_recon = autoencoder.decode(z_proj_tensor)

# predicted_recon_np = predicted_recon.squeeze(0).cpu().numpy()

# ################################################################################
# # 8. Save as NIfTI
# ################################################################################

# out_nifti_path = f"{best_sample['subject_id']}_{best_sample['image_uid']}_predicted_recon.nii.gz"
# nib.save(nib.Nifti1Image(predicted_recon_np, affine=np.eye(4)), out_nifti_path)

# print(f"✅ Saved predicted reconstruction to: {out_nifti_path}")

# # === Load ground-truth follow-up image
# # First find the real path from your metadata
# row_follow_meta = df_metadata[
#     (df_metadata["subject_id"] == best_sample["subject_id"]) &
#     (df_metadata["image_uid"] == best_sample["image_uid"])
# ].iloc[0]

# followup_image_path = row_follow_meta["image_path"]

# # Load real follow-up image
# real_followup_img = nib.load(followup_image_path).get_fdata()

# import matplotlib.pyplot as plt

# # === 1. Load and process real follow-up image
# input_dict = {"image_path": followup_image_path}
# real_followup_processed = transforms_fn(input_dict)["image"].unsqueeze(0).to(DEVICE)  # [B, C, X, Y, Z]

# # === 2. Pass real follow-up through autoencoder
# with torch.no_grad():
#     z_real, _ = autoencoder.encode(real_followup_processed)
#     recon_real = autoencoder.decode(z_real)

# real_recon_np = recon_real.squeeze(0).cpu().numpy()  # (C, X, Y, Z)

# # === 2. Decode starting latent
# # Find the starting scan corresponding to the starting age
# start_row = df_latents[
#     (df_latents["subject_id"] == best_sample["subject_id"]) &
#     (np.isclose(df_latents["age"] * 6, best_sample["start_age"], atol=1e-2))
# ].iloc[0]

# start_latent_vector = start_row[latent_cols].to_numpy()

# # PCA project and inverse to get full latent
# start_latent_full = pca.inverse_transform(pca.transform(start_latent_vector[np.newaxis, :]))[0]
# start_latent_reshaped = start_latent_full.reshape(latent_shape)

# z_start_tensor = torch.tensor(start_latent_reshaped, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# with torch.no_grad():
#     starting_recon = autoencoder.decode(z_start_tensor)

# starting_recon_np = starting_recon.squeeze(0).cpu().numpy()

# # === 3. Decode starting latent (already done earlier, reusing starting_recon_np)

# # === 4. Pick middle slice
# mid_slice = predicted_recon_np.shape[-1] // 2

# starting_slice = starting_recon_np[0, :, :, mid_slice]
# real_slice = real_recon_np[0, :, :, mid_slice]
# predicted_slice = predicted_recon_np[0, :, :, mid_slice]

# # === 5. Plot
# plt.figure(figsize=(18, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(starting_slice, cmap="gray", vmin=0, vmax=1)
# plt.title("Starting Reconstruction")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(real_slice, cmap="gray", vmin=0, vmax=1)
# plt.title("Reconstructed Real Follow-Up")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(predicted_slice, cmap="gray", vmin=0, vmax=1)
# plt.title("Predicted Follow-Up")
# plt.axis("off")

# # === 6. Global title
# start_age = best_sample['start_age']
# followup_age = best_sample['followup_age']
# delta_age = followup_age - start_age
# l2_err = best_sample['l2_error']

# plt.suptitle(
#     f"Subject: {best_sample['subject_id']} | Start Age: {start_age:.2f} | Follow-Up Age: {followup_age:.2f} | ΔAge = {delta_age:.2f} yrs | L2 Error = {l2_err:.2f}",
#     fontsize=16
# )

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # === 7. Save figure
# out_png_path = f"{best_sample['subject_id']}_{best_sample['image_uid']}_recon_comparison_full.png"
# plt.savefig(out_png_path, dpi=150)
# print(f"✅ Saved full comparison figure to: {out_png_path}")

# plt.close()

import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy.interpolate import RBFInterpolator
from itertools import combinations
from collections import defaultdict
from monai import transforms
import matplotlib.pyplot as plt
from brlp import init_autoencoder, const
from sklearn.metrics.pairwise import cosine_similarity

################################################################################
# CONFIG
################################################################################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def compute_cosine_and_norm_similarity(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1)).item()
    norm_sim = min(np.linalg.norm(v1), np.linalg.norm(v2)) / max(np.linalg.norm(v1), np.linalg.norm(v2))
    return cos_sim, norm_sim


def sweep_rbf_hyperparameter(
    param_values,
    param_name,  # 'smoothing' or 'epsilon'
    start_latents,
    delta_vectors,
    test_df,
    df_metadata,
    df_latents,
    latent_shape,
    aekl_checkpoint,
    transforms_fn,
    save_prefix="plots/"
):
    """
    Sweep smoothing or epsilon for RBFInterpolator and plot the effects.
    
    Args:
        param_values (list): Values to try for smoothing or epsilon.
        param_name (str): Either 'smoothing' or 'epsilon'.
        start_latents (np.array): Start latent vectors.
        delta_vectors (np.array): Delta latent vectors.
        test_df (pd.DataFrame): Test set dataframe.
        df_metadata (pd.DataFrame): Metadata dataframe.
        df_latents (pd.DataFrame): Latents dataframe.
        latent_shape (tuple): Shape of latent vector reshaped.
        aekl_checkpoint (str): Path to AE checkpoint.
        transforms_fn: MONAI transforms function.
        save_prefix (str): Base directory to save plots.
    """

    assert param_name in ["smoothing", "epsilon", "neighbors"], f"param_name must be 'smoothing' or 'epsilon', got {param_name}"

    mean_l2_errors = []
    mean_mse_errors = []
    mean_error_diffs = []
    mean_cosine_similarities = []
    mean_norm_similarities = []
    mean_final_similarities = []

    for val in param_values:
        print(f"\n=== Training RBF with {param_name}={val} ===")

        # Train RBF
        rbf = RBFInterpolator(
            start_latents, 
            delta_vectors, 
            kernel='gaussian',
            smoothing=val if param_name == "smoothing" else 1e-2,
            neighbors=val if param_name == "neighbors" else None,
            epsilon=val if param_name == "epsilon" else 1.0
        )

        # Predict on test set
        errors = []

        paired_test = defaultdict(list)
        for _, row in test_df.iterrows():
            paired_test[row["subject_id"]].append(row)

        for subject_id, scans in paired_test.items():
            if len(scans) < 2:
                continue
            scans = sorted(scans, key=lambda r: r["age"])
    
            for i, j in combinations(range(len(scans)), 2):
                row_start = scans[i]
                row_follow = scans[j]
                delta_t = row_follow["age"] * 6 - row_start["age"] * 6
                if delta_t <= 0:
                    continue

                start_latent = row_start[[col for col in test_df.columns if col.startswith("latent_")]].to_numpy()
                true_follow_latent = row_follow[[col for col in test_df.columns if col.startswith("latent_")]].to_numpy()

                delta_pred = rbf(start_latent[np.newaxis, :])[0]
                trajectory_true = true_follow_latent - start_latent
                trajectory_pred = delta_pred * delta_t
                predicted_latent = start_latent + trajectory_pred

                l2 = np.linalg.norm(predicted_latent - true_follow_latent)
                #error diff should be normalized
                error_diff = l2 - np.linalg.norm(trajectory_true)
                mse = np.mean((predicted_latent - true_follow_latent) ** 2)

                cos_sim, norm_sim = compute_cosine_and_norm_similarity(trajectory_pred, trajectory_true)
                final_sim = cos_sim * norm_sim

                errors.append({
                    "subject_id": subject_id,
                    "image_uid": row_follow["image_uid"],
                    "start_age": row_start["age"] * 6,
                    "followup_age": row_follow["age"] * 6,
                    "l2_error": l2,
                    "error_diff": error_diff,
                    "mse": mse,
                    "cosine_similarity": cos_sim,
                    "norm_similarity": norm_sim,
                    "final_similarity": final_sim,
                    "predicted_latent": predicted_latent,
                    "true_followup_latent": true_follow_latent
                })

        error_df = pd.DataFrame(errors)
        # mean_l2_errors.append(error_df["l2_error"].mean())
        # mean_error_diffs.append(error_df["error_diff"].mean())

        # --- Collect delta_t and error_diff ---
        delta_times = [e["followup_age"] - e["start_age"] for e in errors]
        error_diffs = [e["error_diff"] for e in errors]

        # Save mean values
        mean_l2_errors.append(error_df["l2_error"].mean())
        mean_mse_errors.append(error_df["mse"].mean())
        mean_error_diffs.append(error_df["error_diff"].mean())
        mean_cosine_similarities.append(error_df["cosine_similarity"].mean())
        mean_norm_similarities.append(error_df["norm_similarity"].mean())
        mean_final_similarities.append(error_df["final_similarity"].mean())

        print(f"✅ {param_name}={val} | Mean L2 Error: {mean_l2_errors[-1]:.4f} | Mean MSE: {mean_mse_errors[-1]:.4f} | Mean Error Diff: {mean_error_diffs[-1]:.4f} | Mean CosSim: {mean_cosine_similarities[-1]:.4f} | Mean NormSim: {mean_norm_similarities[-1]:.4f}")


        # === Per delta_t scatter plot ===
        plt.figure(figsize=(8, 6))

        # plt.scatter(delta_times, [e["error_diff"] for e in errors], alpha=0.7, label="Error Diff (L2 - norm(traject_true))", marker='s')

        plt.scatter(delta_times, [e["norm_similarity"] for e in errors], alpha=0.7, label="Norm sim (traject_true vs pred)", marker='o')

        plt.scatter(delta_times, [e["cosine_similarity"] for e in errors], alpha=0.7, label="cosine sim (traject_true vs pred)", marker='*')

        plt.scatter(delta_times, [e["final_similarity"] for e in errors], alpha=0.7, label="final sim (cos*norm)", marker='^')
        # Horizontal reference line
        plt.axhline(0, color='red', linestyle='--', label="Zero Reference")
        # Labels and styling
        plt.xlabel("Δ Time (years)")
        plt.ylabel("Error Metric Value")
        plt.title(f"Error Metrics vs Delta Time\n({param_name}={val:.0e})")
        plt.grid(True)
        plt.legend()
        # Save figure
        os.makedirs(f"{save_prefix}/scatter_plots", exist_ok=True)
        scatter_path = f"{save_prefix}/scatter_plots/scatter_{param_name}_{val:.0e}.png".replace("-", "m")
        plt.savefig(scatter_path, dpi=150)
        plt.close()

        print(f"✅ Saved scatter plot {scatter_path}")
        
        # Plot best and worst samples
        best_sample = min(errors, key=lambda x: x["error_diff"])
        worst_sample = max(errors, key=lambda x: x["error_diff"])

        # Load AE
        autoencoder = init_autoencoder(aekl_checkpoint).to(DEVICE).eval()

        param_tag = f"{param_name}_{val:.0e}".replace("-", "m")
        save_dir = os.path.join(save_prefix, param_tag)
        os.makedirs(save_dir, exist_ok=True)

        plot_recon_comparison(
            sample=best_sample,
            df_metadata=df_metadata,
            df_latents=df_latents,
            autoencoder=autoencoder,
            latent_shape=latent_shape,
            transforms_fn=transforms_fn,
            save_dir=save_dir
        )

        plot_recon_comparison(
            sample=worst_sample,
            df_metadata=df_metadata,
            df_latents=df_latents,
            autoencoder=autoencoder,
            latent_shape=latent_shape,
            transforms_fn=transforms_fn,
            save_dir=save_dir
        )

    # === Plot Mean Errors
    plt.figure(figsize=(10, 5))
    # plt.plot(param_values, mean_l2_errors, marker='o', label="Mean L2 Error")
    plt.plot(param_values, mean_error_diffs, marker='s', label="Mean Error Diff")
    plt.plot(param_values, mean_mse_errors, marker='s', label="Mean MSE")
    plt.plot(param_values, mean_cosine_similarities, marker='s', label="Mean cosine sim")
    plt.plot(param_values, mean_norm_similarities, marker='s', label="Mean norm sim")
    plt.plot(param_values, mean_final_similarities, marker='s', label="Mean final sim (cos*norm)")
    plt.xscale('log')
    plt.xlabel(param_name.capitalize())
    plt.ylabel("Error")
    plt.title(f"RBF {param_name.capitalize()} vs Prediction Errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_curve = f"rbf_{param_name}_vs_error.png"
    plt.savefig(out_curve, dpi=150)
    plt.show()

    print(f"✅ Saved curve {out_curve}")


def train_and_predict(start_latents, delta_vectors, test_df, smoothing, epsilon=1.0):
    """
    Train RBFInterpolator and predict on test data.
    """
    print(f"Training RBF with smoothing={smoothing}, epsilon={epsilon}")

    rbf = RBFInterpolator(
        start_latents, 
        delta_vectors, 
        kernel='gaussian',
        epsilon=epsilon,
        smoothing=smoothing
    )

    errors = []

    paired_test = defaultdict(list)
    for _, row in test_df.iterrows():
        paired_test[row["subject_id"]].append(row)

    for subject_id, scans in paired_test.items():
        if len(scans) < 2:
            continue
        scans = sorted(scans, key=lambda r: r["age"])

        for i, j in combinations(range(len(scans)), 2):
            row_start = scans[i]
            row_follow = scans[j]
            delta_t = row_follow["age"] * 6 - row_start["age"] * 6
            if delta_t <= 0:
                continue

            start_latent = row_start[latent_cols].to_numpy()
            true_follow_latent = row_follow[latent_cols].to_numpy()

            delta_pred = rbf(start_latent[np.newaxis, :])[0]
            predicted_latent = start_latent + delta_pred * delta_t

            l2 = np.linalg.norm(predicted_latent - true_follow_latent)
            mse = np.mean((predicted_latent - true_follow_latent) ** 2)
            error_diff = l2 - np.linalg.norm(true_follow_latent - start_latent)

            errors.append({
                "subject_id": subject_id,
                "image_uid": row_follow["image_uid"],
                "start_age": row_start["age"] * 6,
                "followup_age": row_follow["age"] * 6,
                "l2_error": l2,
                "mse": mse,
                "error_diff": error_diff,
                "predicted_latent": predicted_latent,
                "true_followup_latent": true_follow_latent
            })

    return errors


def plot_recon_comparison(sample, df_metadata, df_latents, autoencoder, latent_shape, transforms_fn, save_dir="."):
    """
    Plot and save reconstruction comparison for a given prediction sample.
    
    Args:
        sample (dict): The sample dictionary with fields like 'subject_id', 'image_uid', 'start_age', 'followup_age', etc.
        df_metadata (pd.DataFrame): Metadata CSV containing image paths.
        df_latents (pd.DataFrame): Latent trajectories dataframe.
        autoencoder (torch.nn.Module): Loaded trained autoencoder.
        latent_shape (tuple): Shape of latent tensor (e.g., (3, 14, 18, 14)).
        transforms_fn (monai.transforms.Compose): MONAI preprocessing transforms.
        save_dir (str): Directory to save the figure.
    """
    
    DEVICE = next(autoencoder.parameters()).device

    # === 1. Decode predicted latent
    predicted_latent = np.asarray(sample["predicted_latent"], dtype=np.float32)
    predicted_latent_reshaped = predicted_latent.reshape(latent_shape)
    z_proj_tensor = torch.tensor(predicted_latent_reshaped, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted_recon = autoencoder.decode(z_proj_tensor)

    predicted_recon_np = predicted_recon.squeeze(0).cpu().numpy()

    # === 2. Load and reconstruct real follow-up
    row_follow_meta = df_metadata[
        (df_metadata["subject_id"] == sample["subject_id"]) &
        (df_metadata["image_uid"] == sample["image_uid"])
    ].iloc[0]

    print('follow:', row_follow_meta)

    followup_image_path = row_follow_meta["image_path"]

    input_dict = {"image_path": followup_image_path}
    real_followup_processed = transforms_fn(input_dict)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z_real, _ = autoencoder.encode(real_followup_processed)
        recon_real = autoencoder.decode(z_real)

    real_recon_np = recon_real.squeeze(0).cpu().numpy()

    # === 3. Load and reconstruct starting scan
    start_row = df_latents[
        (df_latents["subject_id"] == sample["subject_id"]) &
        (np.isclose(df_latents["age"] * 6, sample["start_age"], atol=1e-2))
    ].iloc[0]

    print('start:', start_row)

    start_latent_vector = np.asarray(start_row[[col for col in df_latents.columns if col.startswith("latent_")]], dtype=np.float32)
    start_latent_reshaped = start_latent_vector.reshape(latent_shape)

    z_start_tensor = torch.tensor(start_latent_reshaped, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        starting_recon = autoencoder.decode(z_start_tensor)

    starting_recon_np = starting_recon.squeeze(0).cpu().numpy()

    # === 4. Plot middle slices (adjusted)
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

    # Info
    start_age = sample['start_age']
    followup_age = sample['followup_age']
    delta_age = followup_age - start_age
    l2_err = sample['l2_error']
    mse = sample['mse']
    err_diff = sample['error_diff']

    # === Plot nicely
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Bigger figure
    axes = axes.flatten()

    titles = [
        f"Starting Axial | Start Age: {start_age:.2f}",
        f"Real Follow-Up Axial | Follow-Up Age: {followup_age:.2f}",
        "Predicted Follow-Up Axial",
        "Starting Coronal",
        "Real Follow-Up Coronal",
        "Predicted Follow-Up Coronal",
        "Starting Sagittal",
        "Real Follow-Up Sagittal",
        "Predicted Follow-Up Sagittal"
    ]

    images = [
        starting_slice_ax, real_slice_ax, predicted_slice_ax,
        starting_slice_cor, real_slice_cor, predicted_slice_cor,
        starting_slice_sag, real_slice_sag, predicted_slice_sag
    ]

    for ax, img, title in zip(axes, images, titles):
        rotated_img = np.rot90(img, k=1)  # Rotate -90 degrees
        ax.imshow(rotated_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        f"Subject: {sample['subject_id']} | ΔAge = {delta_age:.2f} yrs | L2 Error = {l2_err:.2f} | MSE = {mse:.2f} | Error diff = {err_diff:.2f}",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for main title

    os.makedirs(save_dir, exist_ok=True)
    out_png_path = os.path.join(save_dir, f"{sample['subject_id']}_{sample['image_uid']}_recon_comparison.png")
    plt.savefig(out_png_path, dpi=150)
    plt.close()

    print(f"✅ Saved figure to {out_png_path}")

def plot_delta_norms(delta_age, delta_vectors, save_path=None):
    """
    Plot histogram of delta vector norms.
    
    Args:
        delta_age (np.ndarray): age differences, scalar vector
        delta_vectors (np.ndarray): Delta vectors (N, D)
        save_path (str, optional): Where to save the figure. If None, just show.
    """
    delta_vectors = np.asarray(delta_vectors, dtype=np.float32)
    delta_norms = np.linalg.norm(delta_vectors, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.hist(delta_norms, bins=50, color='steelblue', edgecolor='black')
    plt.title("Histogram of Delta Vector Norms", fontsize=16)
    plt.xlabel("Norm of Delta Vector", fontsize=14)
    plt.ylabel("Number of Pairs", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved delta norms histogram to {save_path}")
    else:
        print('No saving path given')

def plot_age_diff_histogram(age_diffs, save_path=None):
    """
    Plot a histogram of age differences between scan pairs.

    Args:
        age_diffs (list or np.ndarray): List of age differences (in months).
        save_path (str, optional): Where to save the figure. If None, shows the plot.
    """
    age_diffs = np.asarray(age_diffs, dtype=np.float32)

    plt.figure(figsize=(8, 6))
    plt.hist(age_diffs, bins=np.arange(0, 4.5, 0.08), color="skyblue", edgecolor="black")
    plt.title("Histogram of Age Differences Between Scans")
    plt.xlabel("Age Difference (years)")
    plt.ylabel("Number of Pairs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("age_diff_histogram.png", dpi=150)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved age difference histogram to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":

    latent_shape = (3, 14, 18, 14)
    latent_dim = np.prod(latent_shape)

    latent_trajectories_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
    metadata_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
    aekl_checkpoint = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"
    latent_mean_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy"
    latent_std_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"

    # RBF_SMOOTHING = 1e-2

    # MONAI transforms
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    ################################################################################
    # 1. Load data
    ################################################################################
    df_latents = pd.read_csv(latent_trajectories_csv)
    df_metadata = pd.read_csv(metadata_csv)

    latent_cols = [col for col in df_latents.columns if col.startswith("latent_")]

    ################################################################################
    # 2. Build Training Set
    ################################################################################
    train_df = df_latents[df_latents["split"] == "train"]

    paired_train = defaultdict(list)
    for _, row in train_df.iterrows():
        paired_train[row["subject_id"]].append(row)

    start_latents = []
    delta_vectors = []
    age_diffs = []

    for subject_id, scans in paired_train.items():
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

    start_latents = np.vstack(start_latents)
    delta_vectors = np.vstack(delta_vectors)

    print(start_latents.dtype, 'shape', start_latents.shape)
    print(delta_vectors.dtype, 'shape', delta_vectors.shape)

    print(f"Training samples: {start_latents.shape[0]}, Latent dimension: {start_latents.shape[1]}")

    plot_delta_norms(age_diffs, delta_vectors, 'delta_norms_train.png')
    plot_age_diff_histogram(age_diffs, "age_diff_histogram.png")


    test_df = df_latents[df_latents["split"] == "test"] 


    # sweep_rbf_hyperparameter(
    #     param_values=[0.001, 0.01, 0.02, 0.1, 0.2],
    #     param_name="epsilon",
    #     start_latents=start_latents,
    #     delta_vectors=delta_vectors,
    #     test_df=test_df,
    #     df_metadata=df_metadata,
    #     df_latents=df_latents,
    #     latent_shape=latent_shape,
    #     aekl_checkpoint=aekl_checkpoint,
    #     transforms_fn=transforms_fn,
    #     save_prefix="plots/"
    # )

    sweep_rbf_hyperparameter(
        param_values=[500, 614],
        param_name="neighbors",
        start_latents=start_latents,
        delta_vectors=delta_vectors,
        test_df=test_df,
        df_metadata=df_metadata,
        df_latents=df_latents,
        latent_shape=latent_shape,
        aekl_checkpoint=aekl_checkpoint,
        transforms_fn=transforms_fn,
        save_prefix="plots/"
    )
    # sweep_rbf_hyperparameter(
    #     param_values=[5, 10, 50, 100, 150, 200, 300, 400, 500, 614],
    #     param_name="neighbors",
    #     start_latents=start_latents,
    #     delta_vectors=delta_vectors,
    #     test_df=test_df,
    #     df_metadata=df_metadata,
    #     df_latents=df_latents,
    #     latent_shape=latent_shape,
    #     aekl_checkpoint=aekl_checkpoint,
    #     transforms_fn=transforms_fn,
    #     save_prefix="plots/"
    # )

    # sweep_rbf_hyperparameter(
    #     param_values=[0, 1e-4, 1e-3, 1e-2, 1e-1],
    #     param_name="smoothing",
    #     start_latents=start_latents,
    #     delta_vectors=delta_vectors,
    #     test_df=test_df,
    #     df_metadata=df_metadata,
    #     df_latents=df_latents,
    #     latent_shape=latent_shape,
    #     aekl_checkpoint=aekl_checkpoint,
    #     transforms_fn=transforms_fn,
    #     save_prefix="plots/"
    # )


    # ################################################################################
    # # 3. Train RBF Interpolator (Gaussian Kernel)
    # ################################################################################
    # print("Training RBF Interpolator...")
    # rbf = RBFInterpolator(
    #     start_latents, 
    #     delta_vectors, 
    #     kernel='gaussian',
    #     epsilon=1.0,  # Controls width of gaussian; you might want to tune this
    #     smoothing=RBF_SMOOTHING
    # )

    # ################################################################################
    # # 4. Predict on Test Set
    # ################################################################################
    # test_df = df_latents[df_latents["split"] == "test"]

    # paired_test = defaultdict(list)
    # for _, row in test_df.iterrows():
    #     paired_test[row["subject_id"]].append(row)

    # errors = []

    # for subject_id, scans in paired_test.items():
    #     if len(scans) < 2:
    #         continue
    #     scans = sorted(scans, key=lambda r: r["age"])
        
    #     for i, j in combinations(range(len(scans)), 2):
    #         row_start = scans[i]
    #         row_follow = scans[j]
    #         delta_t = row_follow["age"] * 6 - row_start["age"] * 6
    #         if delta_t <= 0:
    #             continue

    #         start_latent = row_start[latent_cols].to_numpy()
    #         true_follow_latent = row_follow[latent_cols].to_numpy()

    #         delta_pred = rbf(start_latent[np.newaxis, :])[0]
    #         predicted_latent = start_latent + delta_pred * 2 * delta_t

            
    #         l2 = np.linalg.norm(predicted_latent - true_follow_latent)
    #         error_diff = l2 - np.linalg.norm(true_follow_latent - start_latent) #should be negative
    #         mse = np.mean((predicted_latent - true_follow_latent) ** 2)

    #         errors.append({
    #             "subject_id": subject_id,
    #             "image_uid": row_follow["image_uid"],
    #             "start_age": row_start["age"] * 6,
    #             "followup_age": row_follow["age"] * 6,
    #             "l2_error": l2,
    #             "error_diff": error_diff,
    #             "mse": mse,
    #             "predicted_latent": predicted_latent,
    #             "true_followup_latent": true_follow_latent
    #         })

    # error_df = pd.DataFrame(errors)
    # print("\nError Summary:")
    # print(error_df.describe())

    # ################################################################################
    # # 5. Pick the Best Sample (Lowest L2 Error)
    # ################################################################################
    # # best_sample = min(errors, key=lambda x: x["l2_error"])

    # # print(f"\n✅ Best sample: {best_sample['subject_id']} | {best_sample['image_uid']} | L2 error: {best_sample['l2_error']:.4f}")

    # best_sample = min(errors, key=lambda x: x["error_diff"])

    # print(f"\n✅ Best sample: {best_sample['subject_id']} | {best_sample['image_uid']} |  Error diff: {best_sample['error_diff']:.4f}")

    # worst_sample = max(errors, key=lambda x: x["error_diff"])

    # print(f"\n✅ Worst sample: {worst_sample['subject_id']} | {worst_sample['image_uid']} |  Error diff: {worst_sample['error_diff']:.4f}")

    # autoencoder = init_autoencoder(aekl_checkpoint).to(DEVICE).eval()

    # # Call for best sample
    # plot_recon_comparison(
    #     sample=best_sample,
    #     df_metadata=df_metadata,
    #     df_latents=df_latents,
    #     autoencoder=autoencoder,
    #     latent_shape=latent_shape,
    #     transforms_fn=transforms_fn,
    #     save_dir="plots/"
    # )

    # # Call for worst sample
    # plot_recon_comparison(
    #     sample=worst_sample,
    #     df_metadata=df_metadata,
    #     df_latents=df_latents,
    #     autoencoder=autoencoder,
    #     latent_shape=latent_shape,
    #     transforms_fn=transforms_fn,
    #     save_dir="plots/"
    # )

