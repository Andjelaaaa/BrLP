# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import wandb
# from datetime import datetime
# from collections import defaultdict
# from itertools import combinations
# from monai import transforms
# from brlp import init_autoencoder, const
# import nibabel as nib

# def load_data(
#     latent_trajectories_csv,
#     metadata_csv,
#     latent_shape=(3, 14, 18, 14)
# ):
#     """
#     Load latent trajectory data and compute training delta vectors.

#     Args:
#         latent_trajectories_csv (str): Path to CSV containing latent trajectories.
#         metadata_csv (str): Path to metadata CSV (not used directly here but could be returned).
#         latent_shape (tuple): Shape of reshaped latent vector (default = (3, 14, 18, 14)).

#     Returns:
#         start_latents (np.ndarray): Array of starting latent vectors [N, latent_dim].
#         delta_vectors (np.ndarray): Corresponding delta vectors [N, latent_dim].
#         age_diffs (np.ndarray): Corresponding age differences [N].
#         df_latents (pd.DataFrame): Full latent DataFrame (for later use).
#         df_metadata (pd.DataFrame): Metadata DataFrame (for later use).
#         latent_cols (list): List of column names used for latents.
#     """
#     latent_dim = np.prod(latent_shape)

#     df_latents = pd.read_csv(latent_trajectories_csv)
#     df_metadata = pd.read_csv(metadata_csv)
#     latent_cols = [col for col in df_latents.columns if col.startswith("latent_")]

#     train_df = df_latents[df_latents["split"] == "train"]
#     paired_train = defaultdict(list)
#     for _, row in train_df.iterrows():
#         paired_train[row["subject_id"]].append(row)

#     start_latents = []
#     delta_vectors = []
#     age_diffs = []

#     for subject_id, scans in paired_train.items():
#         if len(scans) < 2:
#             continue
#         scans = sorted(scans, key=lambda r: r["age"])

#         for i, j in combinations(range(len(scans)), 2):
#             row_start = scans[i]
#             row_follow = scans[j]

#             age_diff = row_follow["age"] * 6 - row_start["age"] * 6
#             if age_diff <= 0:
#                 continue

#             start_latent = row_start[latent_cols].to_numpy()
#             follow_latent = row_follow[latent_cols].to_numpy()
#             delta_vector = (follow_latent - start_latent) / age_diff

#             start_latents.append(start_latent)
#             delta_vectors.append(delta_vector)
#             age_diffs.append(age_diff)

#     start_latents = np.vstack(start_latents).astype(np.float32)
#     delta_vectors = np.vstack(delta_vectors).astype(np.float32)
#     age_diffs = np.array(age_diffs, dtype=np.float32)

#     print(f"✅ Loaded {start_latents.shape[0]} training samples")
#     print(f"Latent dimension: {start_latents.shape[1]}")
    
#     return start_latents, delta_vectors, age_diffs, df_latents, df_metadata, latent_cols

# # === Custom similarity loss ===
# def custom_similarity_loss(delta_pred, delta_true, alpha=0.5):
#     delta_pred_normed = F.normalize(delta_pred, dim=1)
#     delta_true_normed = F.normalize(delta_true, dim=1)

#     cos_loss = 1 - torch.sum(delta_pred_normed * delta_true_normed, dim=1).mean()

#     norm_pred = delta_pred.norm(dim=1)
#     norm_true = delta_true.norm(dim=1)
#     norm_ratio = torch.minimum(norm_pred, norm_true) / torch.maximum(norm_pred, norm_true + 1e-8)
#     norm_loss = 1 - norm_ratio.mean()

#     return alpha * cos_loss + (1 - alpha) * norm_loss

# # === Simple MLP model ===
# class DeltaPredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dims[0]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[1], output_dim)
#         )

#     def forward(self, x):
#         return self.layers(x)

# if __name__ == '__main__':
#     # Generate a timestamped run name
#     run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

#     # === Simulate latent dataset ===
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     # Define latent shape
#     latent_shape = (3, 14, 18, 14)
#     latent_dim = np.prod(latent_shape)

#     # Load the real latent training data
#     latent_trajectories_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
#     metadata_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"

#     start_latents, delta_vectors, age_diffs, df_latents, df_metadata, latent_cols = load_data(
#         latent_trajectories_csv=latent_trajectories_csv,
#         metadata_csv=metadata_csv,
#         latent_shape=latent_shape
#     )

#     follow_latents = start_latents + delta_vectors * age_diffs[:, None]

#     # Initialize wandb
#     wandb.init(
#         project="latent-followup-prediction",
#         name=run_name,
#         config={
#             "model": "MLPRegressor",
#             "latent_dim": latent_dim,
#             "hidden_layers": [256, 128],
#             "learning_rate": 1e-3,
#             "max_iter": 500
#         }
#     )

#     # === Prepare inputs and outputs ===
#     X = np.hstack([start_latents, age_diffs[:, None]])
#     y = follow_latents

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64)

#     # === Initialize model ===
#     input_dim = latent_dim + 1
#     model = DeltaPredictor(input_dim=input_dim, hidden_dims=(256, 128), output_dim=latent_dim)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # === Training loop ===
#     for epoch in range(1, 101):
#         model.train()
#         wandb.log({"train_loss": loss.item(), "epoch": epoch})
#         total_loss = 0
#         for xb, yb in train_loader:
#             pred_latent = model(xb)
#             start_latent = xb[:, :-1]
#             delta_pred = pred_latent - start_latent
#             delta_true = yb - start_latent
#             loss = custom_similarity_loss(delta_pred, delta_true, alpha=0.5)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#     # === Evaluate ===
#     model.eval()
#     with torch.no_grad():
#         x_test_tensor = torch.from_numpy(X_test)
#         y_test_tensor = torch.from_numpy(y_test)
#         y_pred_tensor = model(x_test_tensor)
#         delta_pred = y_pred_tensor - x_test_tensor[:, :-1]
#         delta_true = y_test_tensor - x_test_tensor[:, :-1]

#         mse = F.mse_loss(y_pred_tensor, y_test_tensor).item()
#         l2 = torch.norm(y_pred_tensor - y_test_tensor, dim=1).mean().item()
#         cos_sim = F.cosine_similarity(delta_pred, delta_true, dim=1).mean().item()

#     wandb.log({
#     "Mean Squared Error": mse,
#     "Mean L2 Distance": l2,
#     "Mean Cosine Similarity": cos_sim,
#     })

#     # === Norm plots ===
#     true_norms = torch.norm(y_test_tensor, dim=1).numpy()
#     pred_norms = torch.norm(y_pred_tensor, dim=1).numpy()

#     plt.figure(figsize=(6, 6))
#     sns.scatterplot(x=true_norms, y=pred_norms, alpha=0.6)
#     plt.plot([true_norms.min(), true_norms.max()], [true_norms.min(), true_norms.max()], 'r--', label='Ideal')
#     plt.xlabel("True Follow-Up Latent Norm")
#     plt.ylabel("Predicted Follow-Up Latent Norm")
#     plt.title("Predicted vs True Latent Norms")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig("latent_norms.png")
#     wandb.log({"Latent Norm Scatterplot": wandb.Image("latent_norms.png")})

#     # Create the metrics DataFrame
#     metrics_df = pd.DataFrame({
#         "Mean Squared Error": [mse],
#         "Mean L2 Distance": [l2],
#         "Mean Cosine Similarity": [cos_sim]
#     })

#     # Print to console
#     print("\n=== Evaluation Metrics ===")
#     print(metrics_df.to_string(index=False))

#     # Optionally save to CSV
#     metrics_df.to_csv("evaluation_metrics.csv", index=False)
#     print("✅ Saved evaluation metrics to 'evaluation_metrics.csv'")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from monai import transforms
from brlp import init_autoencoder, const
import nibabel as nib

# === Load data ===
def load_data(latent_trajectories_csv, metadata_csv, latent_shape=(3, 14, 18, 14)):
    latent_dim = np.prod(latent_shape)
    df_latents = pd.read_csv(latent_trajectories_csv)
    df_metadata = pd.read_csv(metadata_csv)
    latent_cols = [col for col in df_latents.columns if col.startswith("latent_")]

    train_df = df_latents[df_latents["split"] == "train"]
    paired_train = defaultdict(list)
    for _, row in train_df.iterrows():
        paired_train[row["subject_id"]].append(row)

    start_latents = []
    delta_vectors = []
    age_diffs = []
    subject_ids = []
    image_uids = []

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
            subject_ids.append(row_follow["subject_id"])
            image_uids.append(row_follow["image_uid"])

    return (np.vstack(start_latents).astype(np.float32),
            np.vstack(delta_vectors).astype(np.float32),
            np.array(age_diffs, dtype=np.float32),
            subject_ids, image_uids,
            df_latents, df_metadata, latent_cols)


# === Custom similarity loss ===
def combined_loss(delta_pred, delta_true, recon_pred, recon_true, alpha=0.5, beta=0.5):
    # Similarity loss (cos + norm)
    delta_pred_normed = F.normalize(delta_pred, dim=1)
    delta_true_normed = F.normalize(delta_true, dim=1)
    cos_loss = 1 - torch.sum(delta_pred_normed * delta_true_normed, dim=1).mean()

    norm_pred = delta_pred.norm(dim=1)
    norm_true = delta_true.norm(dim=1)
    norm_ratio = torch.minimum(norm_pred, norm_true) / torch.maximum(norm_pred, norm_true + 1e-8)
    norm_loss = 1 - norm_ratio.mean()

    similarity_loss = alpha * cos_loss + (1 - alpha) * norm_loss

    # Reconstruction loss
    recon_loss = F.mse_loss(recon_pred, recon_true)

    # Total
    total = beta * similarity_loss + (1 - beta) * recon_loss
    return total, cos_loss.item(), norm_loss.item(), recon_loss.item()



# === MLP ===
class DeltaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.layers(x)

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

if __name__ == '__main__':
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'image_path'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image']),
    transforms.EnsureChannelFirstD(keys=['image']),
    transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
    transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    latent_shape = (3, 14, 18, 14)
    latent_dim = np.prod(latent_shape)

    latent_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
    meta_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
    aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"

    start_latents, delta_vectors, age_diffs, subject_ids, image_uids, df_latents, df_metadata, latent_cols = load_data(latent_csv, meta_csv)
    follow_latents = start_latents + delta_vectors * age_diffs[:, None]

    X = np.hstack([start_latents, age_diffs[:, None]])
    y = follow_latents

    # === Print shapes ===
    print(f"🔹 start_latents shape: {start_latents.shape}")     # (N, latent_dim)
    print(f"🔹 delta_vectors shape: {delta_vectors.shape}")     # (N, latent_dim)
    print(f"🔹 age_diffs shape: {age_diffs.shape}")             # (N,)
    print(f"🔹 follow_latents shape: {follow_latents.shape}")   # (N, latent_dim)
    print(f"🔹 X (input) shape: {X.shape}")                     # (N, latent_dim + 1)
    print(f"🔹 y (target) shape: {y.shape}")                    # (N, latent_dim)

    # Autoencoder
    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = DeltaPredictor(input_dim=latent_dim + 1, hidden_dims=(256, 128), output_dim=latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    wandb.init(project="ML-latent-followup-prediction", name=run_name, mode="offline")

    for epoch in range(1, 101):
        model.train()
        total_loss, total_cos, total_norm, total_recon = 0, 0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred_latent = model(xb)
            start_latent = xb[:, :-1]
            delta_pred = pred_latent - start_latent
            delta_true = yb - start_latent

            recon_true = []
            recon_pred = []

            with torch.no_grad():
                for z_true, z_pred in zip(yb, pred_latent):
                    z_true = z_true.view(1, *latent_shape)
                    z_pred = z_pred.view(1, *latent_shape)

                    r_true = autoencoder.decode(z_true).squeeze(0)
                    r_pred = autoencoder.decode(z_pred).squeeze(0)

                    recon_true.append(r_true)
                    recon_pred.append(r_pred)

                    # Immediately free unused tensors
                    del z_true, z_pred, r_true, r_pred
                    torch.cuda.empty_cache()

            recon_true = torch.stack(recon_true)
            recon_pred = torch.stack(recon_pred)

            loss, cos_loss, norm_loss, recon_loss = combined_loss(delta_pred, delta_true, recon_pred, recon_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cos += cos_loss
            total_norm += norm_loss
            total_recon += recon_loss

        # === Log all losses
        wandb.log({
            "epoch": epoch,
            "train_total_loss": total_loss / len(train_loader),
            "train_cosine_loss": total_cos / len(train_loader),
            "train_norm_loss": total_norm / len(train_loader),
            "train_recon_loss": total_recon / len(train_loader),
        })

        # === Visualize a sample every 10 epochs
        if epoch % 10 == 0:
            xb_cpu = xb.cpu().numpy()
            yb_cpu = yb.cpu().numpy()
            pred_latent_cpu = pred_latent.detach().cpu().numpy()

            sample_dict = {
                "subject_id": subject_ids[0],
                "image_uid": image_uids[0],
                "start_age": xb_cpu[0, -1],
                "followup_age": xb_cpu[0, -1] + age_diffs[0],
                "l2_error": float(np.linalg.norm(pred_latent_cpu[0] - yb_cpu[0])),
                "mse": float(np.mean((pred_latent_cpu[0] - yb_cpu[0]) ** 2)),
                "error_diff": float(np.linalg.norm(pred_latent_cpu[0] - yb_cpu[0]) - np.linalg.norm(yb_cpu[0] - xb_cpu[0, :-1])),
                "predicted_latent": pred_latent_cpu[0]
            }

            recon_pred_np = recon_pred[0].cpu().numpy()
            recon_true_np = recon_true[0].cpu().numpy()

            z_start_np = xb[0, :-1].view(*latent_shape).detach().cpu().numpy()
            z_start_tensor = torch.from_numpy(z_start_np).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                starting_recon = autoencoder.decode(z_start_tensor).squeeze(0).cpu().numpy()

            log_recon_comparison_to_wandb(
                sample=sample_dict,
                starting_recon_np=starting_recon,
                real_recon_np=recon_true_np,
                predicted_recon_np=recon_pred_np,
                step=epoch,
                tag="Training Recons"
            )
        # Only delete recons after visualizing
        # Safe to delete now
        del xb, yb, pred_latent, delta_pred, delta_true, recon_true, recon_pred
        torch.cuda.empty_cache()
        


    print("✅ Training complete.")

    # === Evaluate ===
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.from_numpy(X_test).to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test).to(DEVICE)
        y_pred_tensor = model(x_test_tensor)
        delta_pred = y_pred_tensor - x_test_tensor[:, :-1]
        delta_true = y_test_tensor - x_test_tensor[:, :-1]

        mse = F.mse_loss(y_pred_tensor, y_test_tensor).item()
        l2 = torch.norm(y_pred_tensor - y_test_tensor, dim=1).mean().item()
        cos_sim = F.cosine_similarity(delta_pred, delta_true, dim=1).mean().item()

    wandb.log({
    "Mean Squared Error": mse,
    "Mean L2 Distance": l2,
    "Mean Cosine Similarity": cos_sim,
    })

    # === Norm plots ===
    true_norms = torch.norm(y_test_tensor, dim=1).numpy()
    pred_norms = torch.norm(y_pred_tensor, dim=1).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_norms, pred_norms, alpha=0.6)
    plt.plot([true_norms.min(), true_norms.max()], [true_norms.min(), true_norms.max()], 'r--', label='Ideal')
    plt.xlabel("True Follow-Up Latent Norm")
    plt.ylabel("Predicted Follow-Up Latent Norm")
    plt.title("Predicted vs True Latent Norms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("latent_norms.png")
    wandb.log({"Latent Norm Scatterplot": wandb.Image("latent_norms.png")})

    # Create the metrics DataFrame
    metrics_df = pd.DataFrame({
        "Mean Squared Error": [mse],
        "Mean L2 Distance": [l2],
        "Mean Cosine Similarity": [cos_sim]
    })

    # Print to console
    print("\n=== Evaluation Metrics ===")
    print(metrics_df.to_string(index=False))

    # Optionally save to CSV
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("✅ Saved evaluation metrics to 'evaluation_metrics.csv'")
