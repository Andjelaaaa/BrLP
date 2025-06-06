# train.py
import os
import torch
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from brlp import init_autoencoder, const
from monai.utils import set_determinism
from monai import transforms
from model import DeltaPredictor, combined_loss
from utils import log_recon_comparison_to_wandb, load_data

class IndexedTensorDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.indices = indices  # This should be an array of integers corresponding to global indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # === Set up ===
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    data_path = "/home/andim/projects/def-bedelb/andim/brlp-data"
    aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"

    latent_shape = (3, 14, 18, 14)
    latent_dim = np.prod(latent_shape)

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    # === Load data ===
    latent_csv = os.path.join(data_path, "latent_trajectories.csv")
    meta_csv = os.path.join(data_path, "A.csv")

    start_latents, delta_vectors, age_diffs, start_ages, followup_ages, subject_ids, image_uids, df_latents, df_metadata, latent_cols, splits = load_data(latent_csv, meta_csv)
    follow_latents = start_latents + delta_vectors * age_diffs[:, None]

    X = np.hstack([start_latents, age_diffs[:, None]])
    y = follow_latents

    assert len(splits) == X.shape[0] == y.shape[0], f"Inconsistent data shapes! X: {X.shape}, y: {y.shape}, splits: {splits.shape}"

    train_mask = splits == "train"
    train_indices = np.where(train_mask)[0]  # array of global indices

    X_train, y_train = X[train_mask], y[train_mask]

    train_dataset = IndexedTensorDataset(X_train, y_train, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # === Print shapes ===
    print(f"🔹 start_latents shape: {start_latents.shape}")     # (N, latent_dim)
    print(pd.DataFrame(start_latents).describe())

    print(f"\n🔹 delta_vectors shape: {delta_vectors.shape}")   # (N, latent_dim)
    print(pd.DataFrame(delta_vectors).describe())

    print(f"\n🔹 age_diffs shape: {age_diffs.shape}")           # (N,)
    print(pd.Series(age_diffs).describe())

    print(f"\n🔹 follow_latents shape: {follow_latents.shape}") # (N, latent_dim)
    print(pd.DataFrame(follow_latents).describe())

    print(f"\n🔹 X (input) shape: {X.shape}")                   # (N, latent_dim + 1)
    print(pd.DataFrame(X).describe())

    print(f"\n🔹 y (target) shape: {y.shape}")                  # (N, latent_dim)
    print(pd.DataFrame(y).describe())

    # === Model ===
    model = DeltaPredictor(input_dim=latent_dim + 1, hidden_dims=(256, 128), output_dim=latent_dim).to(DEVICE)
    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Wandb ===
    wandb.init(project="ML-latent-followup-prediction", name=run_name, mode="offline")

    # === Training ===
    for epoch in range(1, 101):
        model.train()
        total_loss, total_cos, total_norm, total_recon = 0, 0, 0, 0

        for xb, yb, global_idx in train_loader:
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

        wandb.log({
            "epoch": epoch,
            "train_total_loss": total_loss / len(train_loader),
            "train_cosine_loss": total_cos / len(train_loader),
            "train_norm_loss": total_norm / len(train_loader),
            "train_recon_loss": total_recon / len(train_loader),
        })

        if epoch % 10 == 0:
            xb_cpu = xb.cpu().numpy()
            yb_cpu = yb.cpu().numpy()
            pred_latent_cpu = pred_latent.detach().cpu().numpy()

            global_id = global_idx[0].item()
            print(f'Logging global_id: {global_id}...')
            
            sample_dict = {
                "subject_id": subject_ids[global_id], 
                "image_uid": image_uids[global_id], 
                "start_age": start_ages[global_id],
                "followup_age": followup_ages[global_id],
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

        del xb, yb, pred_latent, delta_pred, delta_true, recon_true, recon_pred
        torch.cuda.empty_cache()

    os.makedirs(run_name, exist_ok=True)
    torch.save(model.state_dict(), f"{run_name}/trained_model.pt")
    print(f"✅ Model saved to '{run_name}/trained_model.pt'")
