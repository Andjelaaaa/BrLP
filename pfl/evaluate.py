import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import wandb
from brlp import init_autoencoder, const
from monai import transforms
from model import DeltaPredictor
from utils import load_data

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeltaPredictor model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model .pt file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # === Setup ===
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_shape = (3, 14, 18, 14)
    latent_dim = torch.prod(torch.tensor(latent_shape)).item()

    latent_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
    meta_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
    aekl_ckpt = "/home/andim/scratch/brlp/ae_output/autoencoder-ep-318.pth"

    # === Load data ===
    start_latents, delta_vectors, age_diffs, subject_ids, image_uids, df_latents, df_metadata, latent_cols, splits = load_data(latent_csv, meta_csv)
    follow_latents = start_latents + delta_vectors * age_diffs[:, None]

    X = torch.tensor(start_latents, dtype=torch.float32)
    age_diffs_tensor = torch.tensor(age_diffs[:, None], dtype=torch.float32)
    X = torch.cat([X, age_diffs_tensor], dim=1)
    y = torch.tensor(follow_latents, dtype=torch.float32)

    # Apply pre-defined split
    X_test, y_test = X[splits == "test"], y[splits == "test"] 
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=8)

    # === Load model ===
    model = DeltaPredictor(input_dim=latent_dim + 1, hidden_dims=(256, 128), output_dim=latent_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()

    # === Load autoencoder ===
    autoencoder = init_autoencoder(aekl_ckpt).to(DEVICE).eval()

    # === Evaluate ===
    mse_list, l2_list, cos_sim_list = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
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

    final_metrics = {
        "Mean Squared Error": sum(mse_list) / len(mse_list),
        "Mean L2 Distance": sum(l2_list) / len(l2_list),
        "Mean Cosine Similarity": sum(cos_sim_list) / len(cos_sim_list)
    }

    # === Log to wandb ===
    wandb.init(project="ML-latent-followup-prediction", name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}", mode="offline")
    wandb.log(final_metrics)

    # === Scatter plot ===
    y_pred_tensor = torch.cat([model(xb.to(DEVICE)).cpu() for xb, _ in test_loader])
    y_test_tensor = torch.cat([yb for _, yb in test_loader])
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
    plt.savefig("latent_norms_eval.png")
    wandb.log({"Latent Norm Scatterplot Eval": wandb.Image("latent_norms_eval.png")})

    # === Print and Save ===
    print("\n=== Evaluation Metrics ===")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([final_metrics]).to_csv("evaluation_metrics_eval.csv", index=False)
    print("✅ Saved evaluation metrics to 'evaluation_metrics_eval.csv'")
