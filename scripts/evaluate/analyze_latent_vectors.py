import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.cm as cm
# import seaborn as sns

def create_LDA_plot(output_csv_path, output_fig_name):
    # === Load Data ===
    latent_df = pd.read_csv(output_csv_path)

    # === Extract latent features and class labels ===
    latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
    X = latent_df[latent_cols].values
    y = latent_df["diagnosis"].values

    # === Filter out missing or unknown diagnoses
    mask = latent_df["diagnosis"].isin(["CC", "mTBI", "OI"])
    X = X[mask]
    y = y[mask]
    latent_df = latent_df[mask].copy()

    # === LDA Projection ===
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    latent_df["LD1"] = X_lda[:, 0]
    latent_df["LD2"] = X_lda[:, 1]

    # === Plot LDA projection ===
    plt.figure(figsize=(10, 8))
    color_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
    colors = latent_df["diagnosis"].map(color_map).fillna("gray")

    plt.scatter(latent_df["LD1"], latent_df["LD2"], c=colors, s=60, alpha=0.8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[key], markersize=10, label=key)
        for key in color_map
    ]
    plt.legend(handles=legend_elements)

    plt.title("LDA projection of latent space (by diagnosis)")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

    print(f"✅ Saved LDA plot to {output_fig_name}")

def create_PCA_plot(output_csv_path, variable, output_fig_name):

    # === Load Data ===
    latent_df = pd.read_csv(output_csv_path)

    # === Extract and reduce latent space ===
    latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
    X = latent_df[latent_cols].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    latent_df["PC1"] = X_pca[:, 0]
    latent_df["PC2"] = X_pca[:, 1]

    # === Sort for longitudinal plotting ===
    latent_df = latent_df.sort_values(by=["subject_id", variable])

    plt.figure(figsize=(10, 8))

    if variable == "age":
        cmap = cm.get_cmap("viridis")
        values = latent_df["age"].values
        sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=values * 6, cmap=cmap, s=60)

        # Colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label("Age (years * 6)")  # Because of your multiplication
    elif variable == "sex":
        # Color by sex (0=male=blue, 1=female=red)
        color_map = {0: "blue", 1: "red"}
        colors = latent_df["sex"].map(color_map).fillna("gray")
        sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=colors, s=60)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female (1)')
        ]
        plt.legend(handles=legend_elements)

    elif variable == "diagnosis":
        color_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
        colors = latent_df["diagnosis"].map(color_map).fillna("gray")
        sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=colors, s=60)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[key], markersize=10, label=key)
            for key in color_map
        ]
        plt.legend(handles=legend_elements)

    else:
        raise ValueError("Variable must be 'age', 'sex', or 'diagnosis'.")

    # === Plot trajectories for each subject ===
    for sid, group in latent_df.groupby("subject_id"):
        if len(group) > 1:
            plt.plot(group["PC1"], group["PC2"], color="gray", alpha=0.4, linewidth=1)

    plt.title(f"Latent space trajectories (PCA) colored by {variable}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

    print(f"✅ Saved PCA plot to {output_fig_name}")

    
def create_tsne_plot(output_csv_path, variable, output_fig_name):

    # === Load Data ===
    latent_df = pd.read_csv(output_csv_path)

    # === Extract and reduce latent space ===
    latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
    X = latent_df[latent_cols].values

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X)

    latent_df["TSNE1"] = X_tsne[:, 0]
    latent_df["TSNE2"] = X_tsne[:, 1]

    # === Sort for longitudinal plotting ===
    latent_df = latent_df.sort_values(by=["subject_id", variable])

    plt.figure(figsize=(10, 8))

    if variable == "age":
        cmap = cm.get_cmap("viridis")
        values = latent_df["age"].values
        sc = plt.scatter(latent_df["TSNE1"], latent_df["TSNE2"], c=values * 6, cmap=cmap, s=60)

        cbar = plt.colorbar(sc)
        cbar.set_label("Age (years * 6)")

    elif variable == "sex":
        color_map = {0: "blue", 1: "red"}
        colors = latent_df["sex"].map(color_map)
        sc = plt.scatter(latent_df["TSNE1"], latent_df["TSNE2"], c=colors, s=60)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male (0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female (1)')
        ]
        plt.legend(handles=legend_elements)

    elif variable == "diagnosis":
        color_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
        colors = latent_df["diagnosis"].map(color_map)
        sc = plt.scatter(latent_df["PC1"], latent_df["PC2"], c=colors, s=60)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[key], markersize=10, label=key)
            for key in color_map
        ]
        plt.legend(handles=legend_elements)

    else:
        raise ValueError("Variable must be 'age', 'sex', or 'diagnosis'.")

    # === Plot trajectories for each subject ===
    for sid, group in latent_df.groupby("subject_id"):
        if len(group) > 1:
            plt.plot(group["TSNE1"], group["TSNE2"], color="gray", alpha=0.4, linewidth=1)

    plt.title(f"Latent space trajectories (t-SNE) colored by {variable}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

    print(f"✅ Saved t-SNE plot to {output_fig_name}")

def compute_and_save_latent_stats(csv_path, mean_output_path, std_output_path):
    """
    Compute mean and std of latent vectors from a CSV and save them as .npy files.

    Args:
        csv_path (str): Path to CSV containing flattened latent vectors.
        mean_output_path (str): Path to save the latent mean .npy file.
        std_output_path (str): Path to save the latent std .npy file.
    """

    # Load dataframe
    latent_df = pd.read_csv(csv_path)

    # Find latent columns
    latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
    latent_matrix = latent_df[latent_cols].values  # Shape: (n_samples, n_latent_dims)

    # Compute mean and std
    latent_mean = np.mean(latent_matrix, axis=0)
    latent_std = np.std(latent_matrix, axis=0)

    if np.any(latent_std == 0):
        print("⚠️ Warning: Some latent dimensions have zero variance.")

    # Save to .npy files
    np.save(mean_output_path, latent_mean)
    np.save(std_output_path, latent_std)

    print(f"✅ Saved latent mean to: {mean_output_path}")
    print(f"✅ Saved latent std to: {std_output_path}")


def create_latent_csv(csv_paths, output_csv_path):
    """
    Given one or more metadata CSVs, extracts latent vectors from each and merges them into a single CSV.
    
    Args:
        csv_paths (list of str): List of paths to CSV files containing metadata and latent_path entries.
        output_csv_path (str): Path to save the merged output CSV with flattened latent vectors.
    """
    latent_records = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
            latent_path = row.get("latent_path")
            if not isinstance(latent_path, str) or not latent_path.endswith(".npz") or not os.path.exists(latent_path):
                continue

            try:
                latent = np.load(latent_path)["data"].flatten()
            except Exception as e:
                print(f"❌ Failed to load {latent_path}: {e}")
                continue

            record = {
                "subject_id": row["subject_id"],
                "image_uid": row["image_uid"],
                "age": row["age"],
                "sex": row["sex"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
            }
            record.update({f"latent_{i}": val for i, val in enumerate(latent)})
            latent_records.append(record)

    latent_df = pd.DataFrame(latent_records)
    latent_df.to_csv(output_csv_path, index=False)
    print(f"✅ Merged latent vectors saved to: {output_csv_path}")

if __name__ == '__main__':

    # # === CONFIG ===
    # csv_path = "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv"
    # output_csv_path = "/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv"

    # # === CREATE LATENT CSV ===
    # create_latent_csv(csv_path, output_csv_path)
    # create_latent_csv(
    # csv_paths=[
    #     "/home/andim/projects/def-bedelb/andim/brlp-data/A.csv",
    #     "/home/andim/projects/def-bedelb/andim/brlp-data/A_koala.csv"
    # ],
    # output_csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv"
    # )

    ## === PCA ANALYSIS === 
    # create_PCA_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="age", output_fig_name="latent_pca_age.png")
    # create_PCA_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="sex", output_fig_name="latent_pca_sex.png")
    # create_PCA_plot("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", variable="diagnosis", output_fig_name="latent_pca_diagnosis.png")

    ## === LDA ANALYSIS === 
    create_LDA_plot("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", "latent_lda_diagnosis.png")

    ## === TSNE ANALYSIS === 
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="age", output_fig_name="latent_tsne_age.png")
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="sex", output_fig_name="latent_tsne_sex.png")

    ## === COMPUTE MEAN/STD LATENT VECTORS === 
    # compute_and_save_latent_stats(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv",
    # mean_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy",
    # std_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"
    # )





