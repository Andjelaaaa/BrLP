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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def train_and_plot_LDA_CV(csv_path, output_fig_name, n_splits=5):
    # === Load and filter data
    df = pd.read_csv(csv_path)
    valid_diagnoses = ["CC", "mTBI", "OI"]
    df = df[df["diagnosis"].isin(valid_diagnoses)].copy()
    latent_cols = [col for col in df.columns if col.startswith("latent_")]

    X = df[latent_cols].values
    y = df["diagnosis"].values

    # === Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    print(f"📊 Performing {n_splits}-fold stratified cross-validation with solver='svd'")
    misclassified_df = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx} ---")
        print(f"Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")

        # Count per class in train and valid sets
        train_labels = y[train_idx]
        valid_labels = y[valid_idx]

        train_dist = Counter(train_labels)
        valid_dist = Counter(valid_labels)

        print(f"Train class distribution: {dict(train_dist)}")
        print(f"Valid class distribution: {dict(valid_dist)}")

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_valid)
        # Store misclassified samples
        incorrect_mask = y_pred != y_valid
        misclassified_fold = df.iloc[valid_idx[incorrect_mask]].copy()
        misclassified_fold["true_label"] = y_valid[incorrect_mask]
        misclassified_fold["predicted_label"] = y_pred[incorrect_mask]
        misclassified_fold["fold"] = fold_idx
        misclassified_df.append(misclassified_fold)
        acc = accuracy_score(y_valid, y_pred)
        fold_accuracies.append(acc)
        print(f"Fold {fold_idx}: accuracy = {acc:.4f}")

        if fold_idx == n_splits:
            # Save last fold projection for plotting
            df_valid = df.iloc[valid_idx].copy()
            X_lda_valid = lda.transform(X_valid)
            df_valid["LD1"] = X_lda_valid[:, 0]
            df_valid["LD2"] = X_lda_valid[:, 1]
            df_valid["split"] = "valid"

            df_train = df.iloc[train_idx].copy()
            X_lda_train = lda.transform(X_train)
            df_train["LD1"] = X_lda_train[:, 0]
            df_train["LD2"] = X_lda_train[:, 1]
            df_train["split"] = "train"

            plot_df = pd.concat([df_train, df_valid], axis=0)
    # Combine all misclassified rows into one DataFrame
    all_misclassified = pd.concat(misclassified_df, ignore_index=True)

    # Show summary
    print(f"\n❌ Total misclassified samples: {len(all_misclassified)}")
    print(all_misclassified[["subject_id", "true_label", "predicted_label", "fold"]].head())

    print("\n🔍 Breakdown of misclassified samples per fold:")
    for i, fold_df in enumerate(misclassified_df, start=1):
        print(f"\n--- Fold {i} ---")
        print(f"❌ Misclassified: {len(fold_df)}")

        # Age summary
        if "age" in fold_df.columns:
            print("Age Summary:")
            print(fold_df["age"].describe())

        # Sex counts
        if "sex" in fold_df.columns:
            print("Sex Distribution:")
            print(fold_df["sex"].value_counts(dropna=False))

        # Crosstab of sex and true label
        if "sex" in fold_df.columns and "true_label" in fold_df.columns:
            print("Sex vs True Diagnosis:")
            print(pd.crosstab(fold_df["sex"], fold_df["true_label"]))

    print(f"\n✅ Average CV accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    # === Plot LDA projection (last fold only)
    plt.figure(figsize=(10, 8))
    color_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
    marker_map = {"train": "o", "valid": "s"}

    for split in ["train", "valid"]:
        df_split = plot_df[plot_df["split"] == split]
        for diag in valid_diagnoses:
            df_diag = df_split[df_split["diagnosis"] == diag]
            plt.scatter(df_diag["LD1"], df_diag["LD2"],
                        c=color_map[diag],
                        marker=marker_map[split],
                        label=f"{diag} ({split})",
                        alpha=0.7, s=60)

    handles = [Line2D([0], [0], marker=marker_map[sp], color='w', markerfacecolor=color_map[dx], markersize=10,
                      label=f"{dx} ({sp})")
               for dx in valid_diagnoses for sp in ["train", "valid"]]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"LDA projection (svd, fold {n_splits})")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()
    print(f"✅ Saved LDA plot (last fold) to {output_fig_name}")

def train_and_plot_LDA(output_csv_path, output_fig_name):
    # === Load Data ===
    latent_df = pd.read_csv(output_csv_path)

    # === Keep only known diagnoses
    valid_diagnoses = ["CC", "mTBI", "OI"]
    latent_df = latent_df[latent_df["diagnosis"].isin(valid_diagnoses)].copy()

    # === Extract latent columns
    latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]

    # === Split the data
    train_df = latent_df[latent_df["split"] == "train"]
    valid_df = latent_df[latent_df["split"] == "valid"]
    test_df  = latent_df[latent_df["split"] == "test"]

    X_train = train_df[latent_cols].values
    y_train = train_df["diagnosis"].values
    X_valid = valid_df[latent_cols].values
    y_valid = valid_df["diagnosis"].values
    X_test  = test_df[latent_cols].values
    y_test  = test_df["diagnosis"].values

    # === Try multiple solvers with shrinkage where applicable
    solver_options = [
        {"solver": "svd", "shrinkage": None},               # no shrinkage supported
        {"solver": "lsqr", "shrinkage": "auto"},            # regularized
        {"solver": "eigen", "shrinkage": "auto"}            # regularized
    ]

    best_solver = None
    best_shrinkage = None
    best_acc = -1

    for opt in solver_options:
        try:
            lda = LinearDiscriminantAnalysis(
                solver=opt["solver"],
                shrinkage=opt["shrinkage"],
                n_components=2
            )
            lda.fit(X_train, y_train)
            preds = lda.predict(X_valid)
            acc = accuracy_score(y_valid, preds)
            print(f"Validation acc with solver={opt['solver']}, shrinkage={opt['shrinkage']}: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_solver = opt["solver"]
                best_shrinkage = opt["shrinkage"]

        except Exception as e:
            print(f"⚠️  Skipped solver={opt['solver']} shrinkage={opt['shrinkage']}: {e}")

    print(f"\n✅ Best config: solver={best_solver}, shrinkage={best_shrinkage}, acc={best_acc:.4f}")

    # === Retrain on best config
    final_lda = LinearDiscriminantAnalysis(
        solver=best_solver,
        shrinkage=best_shrinkage,
        n_components=2
    )
    final_lda.fit(X_train, y_train)

    y_pred_test = final_lda.predict(X_test)
    print("\nTest performance:")
    print(classification_report(y_test, y_pred_test))

    # === Apply LDA to all data
    X_all = latent_df[latent_cols].values
    X_lda_all = final_lda.transform(X_all)
    latent_df["LD1"] = X_lda_all[:, 0]
    latent_df["LD2"] = X_lda_all[:, 1] if X_lda_all.shape[1] > 1 else 0.0

    # === Plotting
    plt.figure(figsize=(10, 8))
    color_map = {"CC": "green", "mTBI": "orange", "OI": "purple"}
    marker_map = {"train": "o", "valid": "s", "test": "X"}

    for split in ["train", "valid", "test"]:
        df_split = latent_df[latent_df["split"] == split]
        for diagnosis in valid_diagnoses:
            df_diag = df_split[df_split["diagnosis"] == diagnosis]
            plt.scatter(df_diag["LD1"], df_diag["LD2"],
                        c=color_map[diagnosis],
                        marker=marker_map[split],
                        label=f"{diagnosis} ({split})",
                        alpha=0.7, s=60)

    handles = [Line2D([0], [0], marker=marker_map[sp], color='w', markerfacecolor=color_map[dx], markersize=10,
                      label=f"{dx} ({sp})")
               for dx in valid_diagnoses for sp in ["train", "valid", "test"]]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"LDA (solver={best_solver}, shrinkage={best_shrinkage})")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_name)
    plt.close()

    print(f"✅ Saved LDA plot to {output_fig_name}")


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
    # create_LDA_plot("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", "latent_lda_diagnosis.png")
    # train_and_plot_LDA("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", "latent_lda_diagnosis_test_solver.png")
    train_and_plot_LDA_CV("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", "latent_lda_diagnosis_CV.png")
    
    ## === TSNE ANALYSIS ===
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="age", output_fig_name="latent_tsne_age.png")
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="sex", output_fig_name="latent_tsne_sex.png")

    ## === COMPUTE MEAN/STD LATENT VECTORS ===
    # compute_and_save_latent_stats(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv",
    # mean_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy",
    # std_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"
    # )





