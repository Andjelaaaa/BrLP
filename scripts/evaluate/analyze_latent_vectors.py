import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.cm as cm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import plotly.graph_objects as go

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

def analyze_within_between_latent_distances(csv_path, nbr_clusters):
    # ─── 1. LOAD DATA ────────────────────────────────────────────────────────
    # Replace with your actual path
    df = pd.read_csv(csv_path)

    # Remove KOALA participants
    pattern = r"^sub-\d{5,}$"
    # this will keep only rows where subject_id matches sub- followed by ≥5 digits
    df = df[df['subject_id'].str.match(pattern)]

    # Identify latent columns
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    X = df[latent_cols].values
    subjects = df["subject_id"].values
    ages = df["age"].values.astype(float)

    # ─── 2. NORMALIZE ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # ─── 3. K-MEANS & EXTREME VECTORS ───────────────────────────────────────
    k = nbr_clusters  # choose # clusters you like
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_norm)
    centers = kmeans.cluster_centers_

    # distance of each sample to its cluster centroid
    dists = np.linalg.norm(X_norm - centers[labels], axis=1)

    # define extremes as the top 5% farthest from centroid
    pct = 95
    threshold = np.percentile(dists, pct)
    extreme_mask = dists >= threshold
    extremes_df = df[extreme_mask]

    print(f"{extremes_df.shape[0]} samples ≥ {pct}th percentile distance out of {df.shape[0]} samples")

    df["dist"]               = dists
    df["is_latent_extreme"]  = extreme_mask

    # ─── Compute per‐subject scan counts ─────────────────────────────────
    # (counts the number of rows / image_uid per subject)
    df["n_images"] = df.groupby("subject_id")["image_uid"].transform("count")
    print(df.head(10))

     # ─── Plot scatter: number of scans vs. relative distance ──────────────
    plt.figure(figsize=(6,4))
    sc = plt.scatter(
        df["n_images"],
        df["dist"],
        c=df["age"]*6,          # optional: color‐code by age
        cmap="viridis",
        alpha=0.6,
        s=30
    )
    plt.colorbar(sc, label="Age (years)")
    plt.xlabel("Number of scans for that subject")
    plt.ylabel("Relative distance to cluster centroid")
    plt.title("Scan‐count vs. Relative Latent‐Space Distance")
    plt.tight_layout()
    plt.savefig('nbr_scans_vs_dist_to_centroid.png')

    # 1) collapse to one row per subject
    subj_df = (
        df
        .groupby("subject_id")
        .agg(n_images = ("n_images","first"),
            mean_dist = ("dist","mean"),
            median_dist = ("dist","median"))
        .reset_index()
    )

    # 2) scatter plot: #scans vs. mean distance
    plt.figure(figsize=(6,4))
    plt.scatter(subj_df["n_images"], subj_df["mean_dist"], alpha=0.7)
    plt.xlabel("Number of scans per subject")
    plt.ylabel("Mean distance to cluster centroid")
    plt.title("Scan count vs. mean latent‐space distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scan_count_vs_mean_latent_dist.png")

    # 3) compute nonparametric correlation
    rho, pval = spearmanr(subj_df["n_images"], subj_df["mean_dist"])
    print(f"Spearman ρ = {rho:.2f}, p = {pval:.3f}")

    plt.figure(figsize=(8,4))

    sns.boxplot(
        data=subj_df,
        x="n_images",
        y="mean_dist",
        order=sorted(subj_df["n_images"].unique())
    )
    plt.xlabel("Number of scans per subject")
    plt.ylabel("Mean distance to cluster centroid")
    plt.title("Latent‐space stability vs. longitudinal sample size")
    plt.tight_layout()
    plt.savefig("boxplot_scan_group_count.png")

    # ─── 4. WITHIN- vs BETWEEN-SUBJECT DISTANCES ─────────────────────────────
    # Within‐subject
    within_d = []
    for sid, group in df.groupby("subject_id"):
        Xi = scaler.transform(group[latent_cols].values)
        for i in range(len(Xi)):
            for j in range(i + 1, len(Xi)):
                within_d.append(np.linalg.norm(Xi[i] - Xi[j]))

    # Between‐subject (sample a subset if too big)
    between_d = []
    all_X = X_norm
    all_subj = subjects
    n = len(all_X)
    for i in range(n):
        for j in range(i + 1, n):
            if all_subj[i] != all_subj[j]:
                between_d.append(np.linalg.norm(all_X[i] - all_X[j]))
    # (You can random.sample for speed if n is very large)

    # ─── 5. PLOT BOXPLOT COMPARISON ─────────────────────────────────────────
    plt.figure(figsize=(6,4))
    plt.boxplot([within_d, between_d], labels=["Within-Subj", "Between-Subj"])
    plt.ylabel("Euclidean distance")
    plt.title("Within vs Between Subject Latent Distances")
    plt.tight_layout()
    plt.savefig('euclidean_distances_within_between_latent_distances.png')

    # ─── 6. AGE-DIFFERENCE vs LATENT DISTANCE ────────────────────────────────
    age_diff = []
    dist_vals = []
    for sid, group in df.groupby("subject_id"):
        Xi = scaler.transform(group[latent_cols].values)
        ai = group["age"].astype(float).values * 6
        for i in range(len(Xi)):
            for j in range(i + 1, len(Xi)):
                age_diff.append(abs(ai[i] - ai[j]))
                dist_vals.append(np.linalg.norm(Xi[i] - Xi[j]))

    plt.figure(figsize=(6,4))
    plt.scatter(age_diff, dist_vals, alpha=0.6)
    plt.xlabel("Age difference (years)")
    plt.ylabel("Latent‐space distance")
    plt.title("Within‐Subject Distance vs Age Gap")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('age_difference_vs_latent_distance.png')

    # ─── 7. t-SNE VISUALIZATION COLORED BY SUBJECT ───────────────────────────
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(X_norm)

    # map each subject_id to an integer color
    subj_to_idx = {s: i for i, s in enumerate(np.unique(subjects))}
    colors = [subj_to_idx[s] for s in subjects]

    plt.figure(figsize=(6,6))
    plt.scatter(emb[:,0], emb[:,1], c=colors, cmap="tab20", s=15, alpha=0.8)
    plt.title("t-SNE of All Latents (colored by subject)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig('tsne_by_subject.png')

    # ─── 8. SAVE EXTREMES FOR REVIEW ─────────────────────────────────────────
    # extremes_df.to_csv("latents_extremes.csv", index=False)
    # print("Saved extremes to latents_extremes.csv")
    # save just the columns you care about for downstream use
    out_cols = ["subject_id", "image_uid", "age", "sex",
                "dist", "is_latent_extreme"] + latent_cols
    df[out_cols].to_csv(f"latents_extremes_k_{nbr_clusters}.csv", index=False)

def LME_model_latent_age(csv_input):

    # === Load the merged latents + metadata ===
    df = pd.read_csv(csv_input)       # contains latent dims, age, subject_id
    # If needed, merge with vols too, but age and subject_id must be present.
    # df = pd.merge(df, pd.read_csv("healthy_shape_features.csv"), on=["subject_id","image_uid"])

    # Identify latent dimensions
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    
    df["age_c"] = df["age"] - df["age"].mean()
    results = {}

    for dim in latent_cols:
        df[dim] = (df[dim] - df[dim].mean()) / df[dim].std()
        formula = f"{dim} ~ age_c"
        md = smf.mixedlm(formula, df,
                        groups=df["subject_id"],
                        re_formula="~age_c")
        try:
            mdf = md.fit(reml=False,
                        method='bfgs',
                        maxiter=6000,
                        full_output=True,
                        disp=False)
            # check covariance eigenvalues
            eigs = np.linalg.eigvals(mdf.cov_re)
            if np.min(eigs) < 1e-6:
                raise ValueError("singular cov_re")
        except Exception:
            # fallback to random intercept only
            md_simple = smf.mixedlm(formula, df,
                                    groups=df["subject_id"],
                                    re_formula="1")
            mdf = md_simple.fit()
            print(f"Fell back to intercept‐only for {dim}")

        results[dim] = mdf
        print(f"=== Results for {dim} ===")
        print(mdf.summary())
        print("Random effects cov:\n", mdf.cov_re)
        print("-" * 80)

    re = mdf.random_effects
    for subj, coef in re.items():
        intercept = coef['Group']
        slope = coef['age']
        ages = np.linspace(df['age'].min(), df['age'].max(), 10)
        plt.plot(ages, intercept + slope*ages, color='gray', alpha=0.5)
    plt.scatter(df['age'], df[dim], c='C1'); plt.xlabel('Age'); plt.ylabel(dim)
    plt.savefig('LME_individual_latent_trajectories.png')

def plot_extremes_latent_volumes_with_age(latents, vols):
    # Drop age in one so that it remains in the merged version
    vols = vols.drop(columns="age")
    df = pd.merge(latents, vols, on=["subject_id","image_uid"], how="inner")

    # --- 2) Identify and combine volumetric regions (exclude background) ---
    vol_cols = [c for c in df.columns if c.endswith("__volume_mm3") and not c.startswith("background")]
    regions = sorted({c.split("__volume_mm3")[0].replace("left_","").replace("right_","") 
                    for c in vol_cols})

    for region in regions:
        left  = f"left_{region}__volume_mm3"
        right = f"right_{region}__volume_mm3"
        if left in df.columns and right in df.columns:
            df[f"{region}__volume_mm3"] = df[left] + df[right]

    region_cols = [f"{r}__volume_mm3" for r in regions if f"{r}__volume_mm3" in df.columns]

    # === 3) Compute 95th percentile thresholds & stats ===
    dist95 = np.percentile(df["dist"], 95)
    vol_pcts = {col: np.percentile(df[col], [5, 95]) for col in region_cols}
    region_stats = {}
    for col in region_cols:
        low, high = vol_pcts[col]
        is_vol_ext = (df[col] <= low) | (df[col] >= high)
        ct = pd.crosstab(df["is_latent_extreme"], is_vol_ext)
        n_both = int(ct.loc[True, True]) if True in ct.index and True in ct.columns else 0
        n_latent = int(df["is_latent_extreme"].sum())
        region_stats[col] = (n_both, n_latent)

    # --- 4) Build interactive Plotly figure ---
    fig = go.Figure()

    # Initial region
    init_col = region_cols[0]
    low0, high0 = vol_pcts[init_col]
    n_both0, n_latent0 = region_stats[init_col]
    title0 = f"{n_both0} of {n_latent0} latent‐extreme samples are also volume‐extreme ({init_col})"

    # Scatter trace colored by age
    fig.add_trace(go.Scatter(
        x=df["dist"], y=df[init_col],
        mode="markers",
        marker=dict(
            size=8,
            color=df["age"]*6,              # color by age
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Age (years)")
        )
    ))

    # Latent 95th percentile line
    fig.add_trace(go.Scatter(
        x=[dist95, dist95],
        y=[df[init_col].min(), df[init_col].max()],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"Latent distances 95th pct ({dist95:.1f})"
    ))
    # Volume 95th percentile line
    fig.add_trace(go.Scatter(
        x=[df["dist"].min(), df["dist"].max()],
        y=[high0, high0],
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name=f"{init_col} 95th pct ({high0:.0f})"
    ))  

    fig.data[0].marker.colorbar.update({
    "x": 1.05,       # move it further right of the plotting area
    "y": 0.45,        # center vertically
    "len": 0.8,      # adjust height
    "thickness": 20  # adjust width
    })

    steps = []
    for col in region_cols:
        low, high = vol_pcts[col]
        n_both, n_latent = region_stats[col]
        step_title = f"{n_both} of {n_latent} latent‐extreme samples are also volume‐extreme ({col})"

        steps.append(dict(
            method="update",
            args=[
                {
                    # update data
                    "x": [
                        df["dist"],                         # scatter x
                        [dist95, dist95],                   # latent line x
                        [df["dist"].min(), df["dist"].max()]# volume line x
                    ],
                    "y": [
                        df[col],                            # scatter y
                        [df[col].min(), df[col].max()],     # latent line y
                        [high, high]                        # volume line y
                    ],
                    # **update trace names** for legend
                    "name": [
                        "Samples",
                        f"Latent distances 95th pct ({dist95:.1f})",
                        f"{col} 95th pct ({high:.0f})"
                    ]
                },
                {
                    # update layout
                    "yaxis": {"title": col},
                    "title.text": step_title
                }
            ],
            label=col
        ))

    fig.update_layout(
    title_text=title0,
    xaxis_title="Latent‐space distance",
    yaxis_title=init_col,
    sliders=[dict(
        active=0,
        currentvalue={"prefix": "Region: "},
        pad={"t": 50},
        steps=steps
    )]
    )

    # === 5) Save to HTML ===
    fig.write_html(
        "latent_vs_volume_slider_with_age.html",
        include_plotlyjs="cdn",
        full_html=True
    )

def plot_extremes_latent_volumes(latents, vols):
    # Drop age in one so that it remains in the merged version
    vols = vols.drop(columns="age")
    df = pd.merge(latents, vols, on=["subject_id","image_uid"], how="inner")

    # --- 2) Identify and combine volumetric regions (exclude background) ---
    vol_cols = [c for c in df.columns if c.endswith("__volume_mm3") and not c.startswith("background")]
    regions = sorted({c.split("__volume_mm3")[0].replace("left_","").replace("right_","") 
                    for c in vol_cols})

    for region in regions:
        left  = f"left_{region}__volume_mm3"
        right = f"right_{region}__volume_mm3"
        if left in df.columns and right in df.columns:
            df[f"{region}__volume_mm3"] = df[left] + df[right]

    region_cols = [f"{r}__volume_mm3" for r in regions if f"{r}__volume_mm3" in df.columns]

    # === 3) Compute 95th percentile thresholds & stats ===
    dist95 = np.percentile(df["dist"], 95)
    vol_pcts = {col: np.percentile(df[col], [5, 95]) for col in region_cols}
    region_stats = {}
    for col in region_cols:
        low, high = vol_pcts[col]
        is_vol_ext = (df[col] <= low) | (df[col] >= high)
        ct = pd.crosstab(df["is_latent_extreme"], is_vol_ext)
        n_both = int(ct.loc[True, True]) if True in ct.index and True in ct.columns else 0
        n_latent = int(df["is_latent_extreme"].sum())
        region_stats[col] = (n_both, n_latent)

    # --- 4) Build interactive Plotly figure ---
    fig = go.Figure()

    # Initial region
    init_col = region_cols[0]
    low0, high0 = vol_pcts[init_col]
    n_both0, n_latent0 = region_stats[init_col]
    title0 = f"{n_both0} of {n_latent0} latent‐extreme samples are also volume‐extreme ({init_col})"

    # Scatter trace
    fig.add_trace(go.Scatter(
        x=df["dist"], y=df[init_col],
        mode="markers",
        marker=dict(color=df["is_latent_extreme"].map({False: "gray", True: "red"})),
        name="Samples"
    ))

    # Latent 95th percentile line
    fig.add_trace(go.Scatter(
        x=[dist95, dist95],
        y=[df[init_col].min(), df[init_col].max()],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"Latent distances 95th pct ({dist95:.1f})"
    ))
    # Volume 95th percentile line
    fig.add_trace(go.Scatter(
        x=[df["dist"].min(), df["dist"].max()],
        y=[high0, high0],
        mode="lines",
        line=dict(color="blue", dash="dash"),
        name=f"{init_col} 95th pct ({high0:.0f})"
    ))  

    steps = []
    for col in region_cols:
        low, high = vol_pcts[col]
        # boolean masks for this region
        is_vol_ext = (df[col] <= low) | (df[col] >= high)
        is_lat    = df["is_latent_extreme"]
        
        # build a color array: purple if both, red if latent-only, gray otherwise
        colors = np.where(is_lat & is_vol_ext, "purple",
                np.where(is_lat,              "red",    
                        "gray"))

        n_both, n_latent = region_stats[col]
        title = f"{n_both} of {n_latent} latent‐extreme samples are also volume‐extreme ({col})"

        steps.append(dict(
            method="update",
            args=[
                {
                    "x": [
                        df["dist"],                           # scatter
                        [dist95, dist95],                     # latent cutoff line
                        [df["dist"].min(), df["dist"].max()]  # volume cutoff line
                    ],
                    "y": [
                        df[col],                              # scatter
                        [df[col].min(), df[col].max()],       # latent cutoff line
                        [high, high]                          # volume cutoff line
                    ],
                    # update only trace 0’s marker colors
                    "marker.color": [colors, None, None]
                },
                {
                    "yaxis": {"title": col},
                    "title.text": title
                }
            ],
            label=col
        ))

    # Add slider
    fig.update_layout(
        title_text=title0,
        xaxis_title="Latent‐space distance",
        yaxis_title=init_col,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Region: "},
            pad={"t": 50},
            steps=steps
        )]
    )

    # === 5) Save to HTML ===
    fig.write_html(
        "latent_vs_volume_slider_with_age.html",
        include_plotlyjs="cdn",
        full_html=True
    )

def plot_extremes_latent_vols_one_region(latents, vols, region):
    # 2) Merge on subject & image
    df = pd.merge(latents, vols,
                on=["subject_id","image_uid"],
                how="inner")

    # 3) Pick one volume metric and define extremes
    vol_col = region # ex: "left_cerebral_white_matter__volume_mm3"
    low, high = np.percentile(df[vol_col], [5,95])
    df["is_vol_extreme"] = (
        (df[vol_col] <= low)
        | (df[vol_col] >= high)
    )

    # 4) Build a confusion table
    ct = pd.crosstab(df["is_latent_extreme"], df["is_vol_extreme"],
                    rownames=["latent extreme"], colnames=["volume extreme"])
    print(ct)

    # 5) Quantify overlap
    n_both   = ct.loc[True, True]
    n_latent = df["is_latent_extreme"].sum()
    print(f"{n_both} of {n_latent} latent‐extreme samples are also volume‐extreme")

    # 6) Visualize

    # assume df, dist, and high are already defined:
    v95 = np.percentile(df["dist"], 95)
    h95 = high  # your 95th‐percentile volume

    fig, ax = plt.subplots(figsize=(6,4))

    # scatter
    ax.scatter(df["dist"], df[vol_col],
            c=df["is_latent_extreme"].map({False:"gray", True:"red"}),
            alpha=0.6)

    # 95th‐percentile lines
    ax.axvline(v95, linestyle="--", color="red")
    ax.axhline(h95, linestyle="--", color="blue")

    # annotate the lines
    # for the vertical line, place text at the top of the plot
    ax.text(v95, ax.get_ylim()[1],
            f"{v95:.1f}", color="red",
            ha="right", va="bottom")

    # for the horizontal line, place text at the left of the plot
    ax.text(ax.get_xlim()[0], h95,
            f"{h95:.0f}", color="blue",
            ha="left", va="bottom")

    # labels and title
    ax.set_xlabel("Latent‐space distance")
    ax.set_ylabel(vol_col)             # ensure this label is fully visible
    ax.set_title("Latent distance vs. volume")

    # ensure no clipping of labels
    fig.tight_layout()
    plt.savefig(f"{region}_extremes_scatter.png")
    

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
    # train_and_plot_LDA_CV("/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv", "latent_lda_diagnosis_CV.png")
    
    ## === TSNE ANALYSIS ===
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="age", output_fig_name="latent_tsne_age.png")
    # create_tsne_plot("/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv", variable="sex", output_fig_name="latent_tsne_sex.png")

    ## === COMPUTE MEAN/STD LATENT VECTORS ===
    # compute_and_save_latent_stats(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_trajectories_full.csv",
    # mean_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_mean.npy",
    # std_output_path="/home/andim/projects/def-bedelb/andim/brlp-data/latent_std.npy"
    # )

    ## === BETWEEN AND WITHIN SUBJECT DISTANCES ANALYSIS ===
    csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/merged_latents.csv"
    k = 1
    analyze_within_between_latent_distances(csv_path, nbr_clusters=k)

    ## === COMPARE RELATIONSHIP BETWEEN LATENT DISTANCES AND VOLUME MEASURES ===
    # --- 1) Load and merge data ---
    # vols = pd.read_csv(home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv")
    # plot_extremes_latent_vols_one_region('left_cerebral_white_matter__volume_mm3')
    # plot_extremes_latent_volumes_with_age(latents, vols)
    # plot_extremes_latent_volumes(latents, vols)
    
    ## === MIXED EFFECTS MODEL ON LATENT DISTANCES AND AGE ===
    # LME_model_latent_age("latents_extremes.csv")
    