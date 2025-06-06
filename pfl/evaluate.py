import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import wandb
import numpy as np
import nibabel as nib
from brlp import init_autoencoder, const
from model import DeltaPredictor
from utils import load_data, register_and_jacobian_cli

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeltaPredictor model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model .pt file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_shape = (3, 14, 18, 14)
    latent_dim = torch.prod(torch.tensor(latent_shape)).item()

    latent_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
    meta_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
    aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"

    jacobian_dir = "/home/andim/scratch/jacobian_eval_out"
    os.makedirs(jacobian_dir, exist_ok=True)

    start_latents, delta_vectors, age_diffs, subject_ids, image_uids, df_latents, df_metadata, latent_cols, splits = load_data(latent_csv, meta_csv)
    follow_latents = start_latents + delta_vectors * age_diffs[:, None]

    X = torch.tensor(start_latents, dtype=torch.float32)
    age_diffs_tensor = torch.tensor(age_diffs[:, None], dtype=torch.float32)
    X = torch.cat([X, age_diffs_tensor], dim=1)
    y = torch.tensor(follow_latents, dtype=torch.float32)

    X_test, y_test = X[splits == "test"], y[splits == "test"]
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=1)  # must be 1 for per-sample recon/registration

    model = DeltaPredictor(input_dim=latent_dim + 1, hidden_dims=(256, 128), output_dim=latent_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()

    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

    mse_list, l2_list, cos_sim_list = [], [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            y_pred = model(xb)
            delta_pred = y_pred - xb[:, :-1]
            delta_true = yb - xb[:, :-1]

            mse = F.mse_loss(y_pred, yb).item()
            l2 = torch.norm(y_pred - yb, dim=1).mean().item()
            cos_sim = F.cosine_similarity(delta_pred, delta_true, dim=1).mean().item()

            mse_list.append(mse)
            l2_list.append(l2)
            cos_sim_list.append(cos_sim)

            z_true = yb.view(1, *latent_shape)
            z_pred = y_pred.view(1, *latent_shape)

            recon_true = autoencoder.decode(z_true).squeeze(0).cpu().numpy()
            recon_pred = autoencoder.decode(z_pred).squeeze(0).cpu().numpy()

            print(f"recon_pred_np shape: {recon_pred.shape}")  # should be (D, H, W)
            print(f"recon_true_np shape: {recon_true.shape}")

            if min(recon_pred.shape[-3:]) < 4:
                print(f"⚠️ Skipping sample {i}: too small for registration")
                continue

            pred_path = os.path.join(jacobian_dir, f"pred_{i}.nii.gz")
            true_path = os.path.join(jacobian_dir, f"true_{i}.nii.gz")
            nib.save(nib.Nifti1Image(np.squeeze(recon_pred), affine=np.eye(4)), pred_path)
            nib.save(nib.Nifti1Image(np.squeeze(recon_true), affine=np.eye(4)), true_path)

            z_start = xb[:, :-1].view(1, *latent_shape)
            recon_start = autoencoder.decode(z_start).squeeze(0).cpu().numpy()
            print(f"recon_start shape: {recon_start.shape}")

            # Save start_latent recon
            start_path = os.path.join(jacobian_dir, f"start_{i}.nii.gz")
            nib.save(nib.Nifti1Image(np.squeeze(recon_start), affine=np.eye(4)), start_path)

            register_and_jacobian_cli(
                pred_path=pred_path,
                true_path=true_path,
                output_dir=jacobian_dir,
                tag=f"sample_{i}",
                visualize=True
            )

    final_metrics = {
        "Mean Squared Error": np.mean(mse_list),
        "Mean L2 Distance": np.mean(l2_list),
        "Mean Cosine Similarity": np.mean(cos_sim_list)
    }

    wandb.init(project="ML-latent-followup-prediction", name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}", mode="offline")
    wandb.log(final_metrics)

    print("\n=== Evaluation Metrics ===")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([final_metrics]).to_csv("evaluation_metrics_eval.csv", index=False)
    print("✅ Saved evaluation metrics to 'evaluation_metrics_eval.csv'")
