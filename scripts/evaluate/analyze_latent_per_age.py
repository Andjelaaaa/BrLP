import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns

# === CONFIG ===
csv_path = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
output_csv_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories.csv"

# # === LOAD METADATA ===
# df = pd.read_csv(csv_path)
# latent_records = []

# # === LOAD LATENTS AND FLATTEN ===
# for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting latents"):
#     latent_path = row["latent_path"]
#     if not isinstance(latent_path, str) or not latent_path.endswith(".npz") or not os.path.exists(latent_path):
#         continue

#     try:
#         latent = np.load(latent_path)["data"].flatten()
#         print('Latent shape:', latent.shape)
#     except Exception as e:
#         print(f"❌ Failed to load {latent_path}: {e}")
#         continue

#     record = {
#         "subject_id": row["subject_id"],
#         "image_uid": row["image_uid"],
#         "age": row["age"],
#         "split": row["split"],
#         "diagnosis": row["diagnosis"],
#     }
#     record.update({f"latent_{i}": val for i, val in enumerate(latent)})
#     latent_records.append(record)

# # === CREATE DATAFRAME ===
# latent_df = pd.DataFrame(latent_records)
# latent_df.to_csv(output_csv_path, index=False)
# print(f"✅ Latent vectors saved to: {output_csv_path}")

## === TSNE ANALYSIS === 

# # === Load Data ===
# latent_df = pd.read_csv(output_csv_path)

# # === Extract and reduce latent space ===
# latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
# X = latent_df[latent_cols].values

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# latent_df["PC1"] = X_pca[:, 0]
# latent_df["PC2"] = X_pca[:, 1]

# # === Sort for longitudinal plotting ===
# latent_df = latent_df.sort_values(by=["subject_id", "age"])

# # === Setup plot ===
# plt.figure(figsize=(10, 8))
# cmap = cm.get_cmap("viridis")

# # Normalize age to [0,1] for color mapping
# ages = latent_df["age"].values
# # norm_ages = (ages - ages.min()) / (ages.max() - ages.min())

# # === Scatter points with color by age ===
# sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=ages*6, cmap=cmap, s=60)

# # === Plot trajectories for each subject ===
# for sid, group in latent_df.groupby("subject_id"):
#     if len(group) > 1:
#         plt.plot(group["PC1"], group["PC2"], color="gray", alpha=0.4, linewidth=1)

# # === Add colorbar and labels ===
# cbar = plt.colorbar(sc)
# cbar.set_label("Age (normalized)")

# plt.title("Latent space trajectories (PCA)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('latent_trajectories_actual_age.png')


# # === Load Data ===
# latent_df = pd.read_csv(output_csv_path)

# # === Extract and reduce latent space ===
# latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
# X = latent_df[latent_cols].values

# tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
# X_tsne = tsne.fit_transform(X)

# latent_df["TSNE1"] = X_tsne[:, 0]
# latent_df["TSNE2"] = X_tsne[:, 1]

# # === Sort for longitudinal plotting ===
# latent_df = latent_df.sort_values(by=["subject_id", "age"])

# # === Setup plot ===
# plt.figure(figsize=(10, 8))
# cmap = cm.get_cmap("viridis")

# # Normalize age to [0,1] for color mapping
# ages = latent_df["age"].values
# # norm_ages = (ages - ages.min()) / (ages.max() - ages.min())

# # === Scatter points with color by age ===
# sc = plt.scatter(latent_df["TSNE1"], latent_df["TSNE2"], c=ages*6, cmap=cmap, s=60)

# # === Plot trajectories for each subject ===
# for sid, group in latent_df.groupby("subject_id"):
#     if len(group) > 1:
#         plt.plot(group["TSNE1"], group["TSNE2"], color="gray", alpha=0.4, linewidth=1)

# # === Add colorbar and labels ===
# cbar = plt.colorbar(sc)
# cbar.set_label("Age")

# plt.title("Latent space trajectories (t-SNE)")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('latent_tsne_actual_age.png')

# Load your .csv containing flattened latent vectors
latent_df = pd.read_csv(output_csv_path)

latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
latent_matrix = latent_df[latent_cols].values  # Shape: (n_samples, n_latent_dims)

latent_mean = np.mean(latent_matrix, axis=0)  # shape: (n_latent_dims,)
latent_std  = np.std(latent_matrix, axis=0)

if np.any(latent_std == 0):
    print("⚠️ Warning: Some latent dimensions have zero variance.")


np.save("/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy", latent_mean)
np.save("/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy", latent_std)

