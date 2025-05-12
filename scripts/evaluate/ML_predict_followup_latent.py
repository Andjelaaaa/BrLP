import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
from datetime import datetime
from collections import defaultdict
from itertools import combinations


def load_data(
    latent_trajectories_csv,
    metadata_csv,
    latent_shape=(3, 14, 18, 14)
):
    """
    Load latent trajectory data and compute training delta vectors.

    Args:
        latent_trajectories_csv (str): Path to CSV containing latent trajectories.
        metadata_csv (str): Path to metadata CSV (not used directly here but could be returned).
        latent_shape (tuple): Shape of reshaped latent vector (default = (3, 14, 18, 14)).

    Returns:
        start_latents (np.ndarray): Array of starting latent vectors [N, latent_dim].
        delta_vectors (np.ndarray): Corresponding delta vectors [N, latent_dim].
        age_diffs (np.ndarray): Corresponding age differences [N].
        df_latents (pd.DataFrame): Full latent DataFrame (for later use).
        df_metadata (pd.DataFrame): Metadata DataFrame (for later use).
        latent_cols (list): List of column names used for latents.
    """
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

    start_latents = np.vstack(start_latents).astype(np.float32)
    delta_vectors = np.vstack(delta_vectors).astype(np.float32)
    age_diffs = np.array(age_diffs, dtype=np.float32)

    print(f"✅ Loaded {start_latents.shape[0]} training samples")
    print(f"Latent dimension: {start_latents.shape[1]}")
    
    return start_latents, delta_vectors, age_diffs, df_latents, df_metadata, latent_cols

# === Custom similarity loss ===
def custom_similarity_loss(delta_pred, delta_true, alpha=0.5):
    delta_pred_normed = F.normalize(delta_pred, dim=1)
    delta_true_normed = F.normalize(delta_true, dim=1)

    cos_loss = 1 - torch.sum(delta_pred_normed * delta_true_normed, dim=1).mean()

    norm_pred = delta_pred.norm(dim=1)
    norm_true = delta_true.norm(dim=1)
    norm_ratio = torch.minimum(norm_pred, norm_true) / torch.maximum(norm_pred, norm_true + 1e-8)
    norm_loss = 1 - norm_ratio.mean()

    return alpha * cos_loss + (1 - alpha) * norm_loss

# === Simple MLP model ===
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

if __name__ == '__main__':
    # Generate a timestamped run name
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # === Simulate latent dataset ===
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define latent shape
    latent_shape = (3, 14, 18, 14)

    # Load the real latent training data
    latent_trajectories_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"
    metadata_csv = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"

    start_latents, delta_vectors, age_diffs, df_latents, df_metadata, latent_cols = load_data(
        latent_trajectories_csv=latent_trajectories_csv,
        metadata_csv=metadata_csv,
        latent_shape=latent_shape
    )

    follow_latents = start_latents + delta_vectors * age_diffs[:, None]

    # Initialize wandb
    wandb.init(
        project="latent-followup-prediction",
        name=run_name,
        config={
            "model": "MLPRegressor",
            "latent_dim": latent_dim,
            "hidden_layers": [256, 128],
            "learning_rate": 1e-3,
            "max_iter": 500
        }
    )

    # === Prepare inputs and outputs ===
    X = np.hstack([start_latents, age_diffs[:, None]])
    y = follow_latents

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # === Initialize model ===
    input_dim = latent_dim + 1
    model = DeltaPredictor(input_dim=input_dim, hidden_dims=(256, 128), output_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Training loop ===
    for epoch in range(1, 101):
        model.train()
        wandb.log({"train_loss": loss.item(), "epoch": epoch})
        total_loss = 0
        for xb, yb in train_loader:
            pred_latent = model(xb)
            start_latent = xb[:, :-1]
            delta_pred = pred_latent - start_latent
            delta_true = yb - start_latent
            loss = custom_similarity_loss(delta_pred, delta_true, alpha=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # === Evaluate ===
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test)
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
    sns.scatterplot(x=true_norms, y=pred_norms, alpha=0.6)
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
