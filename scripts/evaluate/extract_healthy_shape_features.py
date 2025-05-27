import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import numpy as np
import scipy.ndimage
# from skimage import measure
from sklearn.decomposition import PCA
from brlp import init_autoencoder, const
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import f_oneway
from pygam import LinearGAM, s
import pickle
import plotly.io as pio
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def compare_diagnosis_vs_control(csv_path, output_csv="group_diff_stats.csv", test="ttest"):
    df = pd.read_csv(csv_path)

    # Identify all z-score columns
    zscore_cols = [col for col in df.columns if "__" in col and col.endswith("_zlocal")]
    diagnoses = df["diagnosis"].unique()
    diagnoses = [d for d in diagnoses if d != "CC"]  # exclude control

    results = []

    for col in zscore_cols:
        region, metric = col.split("__")[0], col.split("__")[1].replace("_zlocal", "")

        cc_values = df[df["diagnosis"] == "CC"][col].dropna()

        for diag in diagnoses:
            group_values = df[df["diagnosis"] == diag][col].dropna()

            if len(cc_values) < 3 or len(group_values) < 3:
                continue  # not enough data

            # Use Welch’s t-test (unequal variance)
            stat, p_value = ttest_ind(group_values, cc_values, equal_var=False)

            results.append({
                "region": region,
                "metric": metric,
                "comparison": f"{diag} vs CC",
                "n_CC": len(cc_values),
                "n_group": len(group_values),
                "group_mean": np.mean(group_values),
                "cc_mean": np.mean(cc_values),
                "mean_diff": np.mean(group_values) - np.mean(cc_values),
                "t_stat": stat,
                "p_value": p_value
            })

    # Adjust p-values (FDR)
    if results:
        p_vals = [r["p_value"] for r in results]
        reject, pvals_corr, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, r in enumerate(results):
            r["p_fdr"] = pvals_corr[i]
            r["significant"] = reject[i]

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"✅ Saved results to {output_csv}")
        return df_results
    else:
        print("⚠️ No valid comparisons made (possibly too few samples).")
        return pd.DataFrame()


# === BASE NAMES (ignoring left/right) ===
def get_base_name(region):
    if region.startswith("left_"):
        return region.replace("left_", "")
    elif region.startswith("right_"):
        return region.replace("right_", "")
    else:
        return region
        
def compute_residual_std_comparison_per_metric(
    csv_all,
    output_dir,
    window_months=3,
    step=1,
    lam=0.8,
    n_splines=15
):
    os.makedirs(output_dir, exist_ok=True)
    print('Starting to plot differences residual std vs local std...')
    df_all = pd.read_csv(csv_all)
    df_all = df_all[df_all["subject_id"] != "sub-3024"]
    df_all = df_all.dropna(subset=["age"])
    df_all["age_months"] = df_all["age"] * 6 * 12
    df_healthy = df_all[df_all["diagnosis"] == "CC"].copy()

    metric_cols = [col for col in df_all.columns if "__" in col and not col.startswith("background")]
    regions = sorted(set(col.split("__")[0] for col in metric_cols))
    metrics = sorted(set(col.split("__")[1] for col in metric_cols))

    for metric in metrics:
        fig = go.Figure()
        visible_traces = []

        for region in regions:
            col = f"{region}__{metric}"
            if col not in df_healthy.columns:
                continue

            try:
                X = df_healthy["age_months"].values.reshape(-1, 1)
                y = df_healthy[col].values
                gam = LinearGAM(s(0), lam=lam, n_splines=n_splines).fit(X, y)
                df_healthy["resid_tmp"] = y - gam.predict(X)

                global_std = df_healthy["resid_tmp"].std()

                ages = np.arange(df_healthy["age_months"].min(), df_healthy["age_months"].max(), step)
                stds = []
                for age in ages:
                    mask = (df_healthy["age_months"] >= age - window_months) & (df_healthy["age_months"] <= age + window_months)
                    window_resids = df_healthy.loc[mask, "resid_tmp"]
                    stds.append(window_resids.std() if len(window_resids) >= 5 else np.nan)

                # Add local std trace
                fig.add_trace(go.Scatter(
                    x=ages,
                    y=stds,
                    mode='lines',
                    name=f"{region} (local)",
                    visible=False
                ))
                idx_local = len(fig.data) - 1

                # Add global std trace
                fig.add_trace(go.Scatter(
                    x=ages,
                    y=[global_std] * len(ages),
                    mode='lines',
                    name=f"{region} (global)",
                    line=dict(dash='dash'),
                    visible=False
                ))
                idx_global = len(fig.data) - 1

                visible_traces.append((region, idx_local, idx_global))

            except Exception as e:
                print(f"⚠️ Failed for {col}: {e}")

        # Create slider steps
        steps = []
        for region, idx_local, idx_global in visible_traces:
            vis = [False] * len(fig.data)
            vis[idx_local] = True
            vis[idx_global] = True
            steps.append(dict(
                method="update",
                args=[{"visible": vis},
                      {"title": f"Residual Std vs Age — {metric}"}],
                label=region
            ))

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Region: "},
            pad={"t": 50},
            steps=steps
        )]

        # Set initial visibility
        if visible_traces:
            first_local, first_global = visible_traces[0][1], visible_traces[0][2]
            fig.data[first_local].visible = True
            fig.data[first_global].visible = True

        fig.update_layout(
            sliders=sliders,
            title=f"Residual Std vs Age — {metric}",
            xaxis_title="Age (months)",
            yaxis_title="Residual Std",
            height=600
        )

        html_path = os.path.join(output_dir, f"resid_std_comparison_{metric}_{window_months}.html")
        pio.write_html(fig, file=html_path, auto_open=False)

    return f"✅ Created {len(metrics)} comparative Plotly HTML files in {output_dir}"

def compute_local_zscores_with_gam(
    csv_all,
    output_csv,
    model_dir,
    window_months=3,
    lam=0.8,
    n_splines=15
    ):

    os.makedirs(model_dir, exist_ok=True)

    df_all = pd.read_csv(csv_all)
    df_all = df_all[df_all["subject_id"] != "sub-3024"]  # optional exclusion
    df_all = df_all.dropna(subset=["age"])
    df_all["age_months"] = df_all["age"] * 12 * 6  # assuming age in years

    df_healthy = df_all[df_all["diagnosis"] == "CC"].copy()

    metric_cols = [col for col in df_healthy.columns if '__' in col and not col.startswith("background")]
    regions = sorted(set(col.split('__')[0] for col in metric_cols))
    metrics = sorted(set(col.split('__')[1] for col in metric_cols))

    # Base columns
    zscore_df = df_all[["subject_id", "image_uid", "age", "diagnosis"]].copy()
    zscore_columns = {}

    print('Starting local z-scoring GAM loop...')
    for metric in tqdm(metrics, desc="Local z-scoring"):
        for region in regions:
            col = f"{region}__{metric}"
            if col not in df_all.columns:
                continue

            try:
                X_healthy = df_healthy["age_months"].values.reshape(-1, 1)
                y_healthy = df_healthy[col].values

                if np.isnan(X_healthy).any() or np.isinf(X_healthy).any():
                    continue
                if np.isnan(y_healthy).any() or np.isinf(y_healthy).any():
                    continue

                gam = LinearGAM(s(0), lam=lam, n_splines=n_splines).fit(X_healthy, y_healthy)

                with open(os.path.join(model_dir, f"{region}__{metric}.pkl"), "wb") as f:
                    pickle.dump(gam, f)

                df_all[f"{col}__pred"] = gam.predict(df_all["age_months"].values.reshape(-1, 1))
                df_all[f"{col}__resid"] = df_all[col] - df_all[f"{col}__pred"]

                z_scores = []
                for i, row in df_all.iterrows():
                    age_i = row["age_months"]
                    resid_i = row[f"{col}__resid"]

                    mask = (
                        (df_healthy["age_months"] >= age_i - window_months) &
                        (df_healthy["age_months"] <= age_i + window_months)
                    )
                    local_resids = df_healthy.loc[mask, col] - gam.predict(df_healthy.loc[mask, "age_months"].values.reshape(-1, 1))

                    if len(local_resids) >= 5 and local_resids.std() > 0:
                        z_i = resid_i / local_resids.std()
                    else:
                        z_i = np.nan

                    z_scores.append(z_i)

                zscore_columns[f"{region}__{metric}_zlocal"] = z_scores

            except Exception as e:
                print(f"⚠️ Failed for {col}: {e}")

    # Final merge
    zscore_df = pd.concat([zscore_df, pd.DataFrame(zscore_columns)], axis=1)
    zscore_df.to_csv(output_csv, index=False)
    print(f"✅ Local z-scored data saved to {output_csv}")


def build_slider_steps(fig, regions_metrics):
    num_traces = len(fig.data)
    traces_per_region_metric = 2  # Adjust if you're adding more traces per key

    steps = []

    for i, key in enumerate(regions_metrics):
        visible = [False] * num_traces

        # Each region__metric group gets two traces: data and fit line
        data_idx = i * traces_per_region_metric
        fit_idx = data_idx + 1

        if data_idx < num_traces:
            visible[data_idx] = True
        if fit_idx < num_traces:
            visible[fit_idx] = True

        step = dict(
            method="update",
            label=key,
            args=[{"visible": visible},
                  {"title": f"GAM Fit for {key}"}]
        )
        steps.append(step)

    return steps

def compute_and_plot_gams_from_shape_features(
    csv_all,
    output_csv,
    model_dir,
    plot_html,
    lam=0.8,
    n_splines=15
):
    os.makedirs(model_dir, exist_ok=True)

    df_all = pd.read_csv(csv_all)
    df_all = df_all[df_all["subject_id"] != "sub-3024"]  # optional
    df_all = df_all.dropna(subset=["age"])  # drop missing age
    df_all["age_months"] = df_all["age"] * 6 * 12
    df_healthy = df_all[df_all["diagnosis"] == "CC"].copy()

    metric_cols = [col for col in df_healthy.columns if '__' in col and not col.startswith("background")]
    regions = sorted(set(col.split('__')[0] for col in metric_cols))
    metrics = sorted(set(col.split('__')[1] for col in metric_cols))

    zscore_df = df_all[["subject_id", "image_uid", "age", "age_months", "sex", "diagnosis"]].copy()
    plot_data = []

    for metric in tqdm(metrics, desc="Processing metrics"):
        for region in regions:
            col = f"{region}__{metric}"
            if col not in df_healthy.columns or col not in df_all.columns:
                continue

            X_healthy = (df_healthy["age_months"].values * 6).reshape(-1, 1)  
            y_healthy = df_healthy[col].values

            X_all = (df_all["age_months"].values * 6).reshape(-1, 1) 
            y_all = df_all[col].values

            try:
                print('Trying redisuals GAM z-scoring...')
                # Sanity check before fitting GAM
                if np.isnan(X_healthy).any() or np.isinf(X_healthy).any():
                    print(f"⚠️ Skipping {col}: X_healthy contains NaN or Inf")
                    continue
                if np.isnan(y_healthy).any() or np.isinf(y_healthy).any():
                    print(f"⚠️ Skipping {col}: y_healthy contains NaN or Inf")
                    continue

                print(f"🔍 Checking {col} - NaNs in age: {df_healthy['age_months'].isnull().sum()}, Infs in age: {np.isinf(df_healthy['age_months']).sum()}")
                print(f"NaNs in y: {np.isnan(y_healthy).sum()}, Infs in y: {np.isinf(y_healthy).sum()}")

                # Fit GAM on healthy data
                gam = LinearGAM(s(0), lam=lam, n_splines=n_splines).fit(X_healthy, y_healthy)
                with open(os.path.join(model_dir, f"{region}__{metric}.pkl"), "wb") as f:
                    pickle.dump(gam, f)

                # Predict on both sets
                y_pred_all = gam.predict(X_all)
                y_pred_healthy = gam.predict(X_healthy)

                # Compute healthy residuals
                residuals_healthy = y_healthy - y_pred_healthy
                mean_resid = residuals_healthy.mean()
                std_resid = residuals_healthy.std()

                print(f"{col}: std_resid type = {type(std_resid)}")

                # Compute z-scores for all subjects
                residuals_all = y_all - y_pred_all
                if isinstance(std_resid, (float, int)) and std_resid > 0:
                    z_scores = (residuals_all - mean_resid) / std_resid
                    zscore_df[f"{region}__{metric}_z"] = z_scores
                else:
                    print(f"⚠️ Skipping {col}: invalid std_resid = {std_resid}")

                # Store data for plotting
                x_vals = np.linspace(df_all["age_months"].min(), df_all["age_months"].max(), 100)
                y_vals = gam.predict(x_vals)
                for diagnosis in df_all["diagnosis"].unique():
                    df_diag = df_all[df_all["diagnosis"] == diagnosis]
                    plot_data.append(dict(
                        region=region,
                        metric=metric,
                        x=df_diag["age_months"].values,
                        y=df_diag[col].values,
                        diagnosis=diagnosis,
                        fit_x=x_vals,
                        fit_y=y_vals
                    ))
            except Exception as e:
                print(f"⚠️ GAM failed for {col}: {e}")

    zscore_df.to_csv(output_csv, index=False)

    # Create Plotly plot
    diagnosis_colors = {
    "CC": "green",
    "mTBI": "orange",
    "OI": "purple"
    }

    fig = go.Figure()
    regions_metrics = sorted(set(f"{d['region']}__{d['metric']}" for d in plot_data))
    visibility_map = {key: [False]*len(regions_metrics)*2 for key in regions_metrics}

    for idx, key in enumerate(regions_metrics):
        data_subset = [d for d in plot_data if f"{d['region']}__{d['metric']}" == key]

        # Concatenate all points for this region__metric across diagnosis
        x_all = np.concatenate([d["x"] for d in data_subset])
        y_all = np.concatenate([d["y"] for d in data_subset])
        color_all = np.concatenate([[diagnosis_colors[d["diagnosis"]]] * len(d["x"]) for d in data_subset])

        visibility_map[key][2 * idx] = True
        visibility_map[key][2 * idx + 1] = True

        # Points with diagnosis color
        fig.add_trace(go.Scatter(
            x=x_all, y=y_all,
            mode='markers',
            name=f"{key} (colored by diagnosis)",
            marker=dict(color=color_all, size=5),
            visible=False,
            showlegend=True
        ))

        # Line fit (use first item, all fits are the same)
        d_fit = data_subset[0]
        fig.add_trace(go.Scatter(
            x=d_fit["fit_x"], y=d_fit["fit_y"],
            mode='lines',
            name=f"{key} fit",
            line=dict(dash='dash', color='black'),
            visible=False,
            showlegend=False
        ))

    # Slider to switch between region__metric combinations
    steps = build_slider_steps(fig, regions_metrics)

    assert len(steps) * 2 <= len(fig.data), "Mismatch between steps and trace count"

    fig.update_layout(
        sliders=[dict(active=0, pad={"t": 50}, steps=steps)],
        title="GAM Fits per Region/Metric",
        xaxis_title="Age (years)",
        yaxis_title="Metric Value"
    )
    fig.write_html(plot_html)
    print(f"✅ Z-scored data saved to {output_csv}")
    print(f"📊 Interactive plot saved to {plot_html}")

def calculate_z_scores(df_healthy, df_diagnoses, metrics):
    # Identify regions
    region_names = sorted(set(col.split("__")[0] for col in df_healthy.columns if "__" in col))

    # === 1️⃣ Merge healthy subjects and CC from diagnoses ===
    df_controls_from_diag = df_diagnoses[df_diagnoses["diagnosis"] == "CC"]
    df_all_healthy = pd.concat([df_healthy, df_controls_from_diag], ignore_index=True)

    # === 2️⃣ Compute healthy mean & std using combined healthy data ===
    metric_cols = [f"{region}__{metric}" for region in region_names for metric in metrics]
    healthy_mean = df_all_healthy[metric_cols].mean()
    healthy_std = df_all_healthy[metric_cols].std()

    # === 3️⃣ Compute z-scores for all diagnosis data ===
    zscores = (df_diagnoses[metric_cols] - healthy_mean) / healthy_std
    zscores["diagnosis"] = df_diagnoses["diagnosis"]
    zscores["subject_id"] = df_diagnoses["subject_id"]

    return region_names, zscores, metric_cols

def compute_long_df_and_analyze(healthy_csv, diagnoses_csv, metrics):
    # === 1️⃣ Load data ===
    df_healthy = pd.read_csv(healthy_csv)
    df_diagnoses = pd.read_csv(diagnoses_csv)

    region_names, zscores, _ = calculate_z_scores(df_healthy, df_diagnoses, metrics)

    # === 4️⃣ Create long-form dataframe ===
    long_records = []
    for _, row in zscores.iterrows():
        for region in region_names:
            for metric in metrics:
                col = f"{region}__{metric}"
                long_records.append({
                    "subject_id": row["subject_id"],
                    "diagnosis": row["diagnosis"],
                    "region": region,
                    "metric": metric,
                    "zscore": row[col] if col in row else np.nan
                })

    long_df = pd.DataFrame(long_records)

    # === 5️⃣ Run statistical analysis ===
    print("Running statistical analysis...")
    summary_df = []

    for metric in metrics:
        print(f"\n===== Metric: {metric} =====")
        for region in region_names:
            sub_df = long_df[(long_df["region"] == region) & (long_df["metric"] == metric)]

            group_values = {}
            for diag in sub_df["diagnosis"].unique():
                group_z = sub_df[sub_df["diagnosis"] == diag]["zscore"].dropna()
                group_values[diag] = group_z

            # Compute group means and stds
            stats = {diag: (group_z.mean(), group_z.std(), len(group_z)) 
                     for diag, group_z in group_values.items()}

            # ANOVA if at least two groups
            if sum(len(g) > 1 for g in group_values.values()) >= 2:
                try:
                    F, p = f_oneway(*[v for v in group_values.values() if len(v) > 1])
                except:
                    p = np.nan
            else:
                p = np.nan

            # Print
            print(f"Region: {region}")
            for diag, (mean_val, std_val, count) in stats.items():
                print(f"  {diag}: mean={mean_val:.3f}, std={std_val:.3f}, n={count}")
            print(f"  ANOVA p-value: {p:.4e}\n")

            row = {"metric": metric, "region": region, "ANOVA_p_value": p}
            for diag in stats:
                mean_val, std_val, count = stats[diag]
                row[f"{diag}_mean"] = mean_val
                row[f"{diag}_std"] = std_val
                row[f"{diag}_n"] = count

            summary_df.append(row)

    summary_df = pd.DataFrame(summary_df)

    print("\n======= Summary Table =======")
    print(summary_df)

    return long_df, summary_df

def plot_spider_charts_zlocal(csv_path, output_prefix="zscore_shape_spider_chart_zlocal"):
    df = pd.read_csv(csv_path)

    # Infer region names and metrics from columns
    zlocal_cols = [col for col in df.columns if col.endswith("_zlocal")]
    regions = sorted(set(col.split("__")[0] for col in zlocal_cols))
    metrics = sorted(set(col.split("__")[1].replace("_zlocal", "") for col in zlocal_cols))

    base_region_names = sorted(set(get_base_name(r) for r in regions))
    px_colors = px.colors.qualitative.Plotly
    base_region_colors = {base: px_colors[i % len(px_colors)] for i, base in enumerate(base_region_names)}
    region_colors = {region: base_region_colors[get_base_name(region)] for region in regions}

    # Compute global max across all z-local medians for scaling
    all_median_values = []
    for diag in ["CC", "mTBI", "OI"]:
        diag_data = df[df["diagnosis"] == diag]
        for region in regions:
            region_metric_cols = [f"{region}__{metric}_zlocal" for metric in metrics]
            if all(col in diag_data.columns for col in region_metric_cols):
                median_abs_z = diag_data[region_metric_cols].abs().median()
                all_median_values.append(median_abs_z.values)
    global_rmax = np.max(np.vstack(all_median_values)) if all_median_values else 1

    # Plot one spider chart per diagnosis
    results = {}
    for diag in ["CC", "mTBI", "OI"]:
        diag_data = df[df["diagnosis"] == diag]
        fig = go.Figure()

        for region in regions:
            region_metric_cols = [f"{region}__{metric}_zlocal" for metric in metrics]
            if not all(col in diag_data.columns for col in region_metric_cols):
                continue

            median_abs_z = diag_data[region_metric_cols].abs().median()

            fig.add_trace(go.Scatterpolar(
                r=median_abs_z.values,
                theta=metrics,
                fill='toself',
                name=region,
                line=dict(color=region_colors[region])
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    title="|Median Local Z-score|",
                    range=[0, global_rmax]
                )
            ),
            showlegend=True,
            title=f"Local Z-Score Deviation per Region ({diag})"
        )

        out_html = f"{output_prefix}_{diag}.html"
        fig.write_html(out_html)
        results[diag] = out_html

    return results
        
def plot_spider_charts_per_diagnosis(
    healthy_csv,
    diagnoses_csv,
    output_prefix="zscore_shape_spider_chart"
    ):

    # === Load Data ===
    df_healthy = pd.read_csv(healthy_csv)
    df_diagnoses = pd.read_csv(diagnoses_csv)

    # === Metric categories ===
    metrics = [
        "volume_mm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "surface_area_mm2",
        "compactness",
        "elongation"
    ]

    region_names, zscores, _ = calculate_z_scores(df_healthy, df_diagnoses, metrics)

    print(f"Found {len(region_names)} regions.")

    base_region_names = sorted(set(get_base_name(r) for r in region_names))

    # === 🎨 Define color palette by base region ===
    px_colors = px.colors.qualitative.Plotly
    base_region_colors = {base: px_colors[i % len(px_colors)] for i, base in enumerate(base_region_names)}
    region_colors = {region: base_region_colors[get_base_name(region)] for region in region_names}

    # === 4️⃣ Find global max for normalization across all diagnoses ===
    all_median_values = []
    for diag in ["CC", "mTBI", "OI"]:
        diag_data = zscores[zscores["diagnosis"] == diag]
        for region in region_names:
            region_metric_cols = [f"{region}__{metric}" for metric in metrics]
            if all(col in diag_data.columns for col in region_metric_cols):
                median_abs_z = diag_data[region_metric_cols].abs().median()
                all_median_values.append(median_abs_z.values)

    if all_median_values:
        global_rmax = np.max(np.vstack(all_median_values))
    else:
        global_rmax = 1  # fallback

    print(f"✅ Global radial max across all groups: {global_rmax:.2f}")

    # === 5️⃣ Plot spider chart per diagnosis ===
    diagnoses = ["CC", "mTBI", "OI"]

    for diag in diagnoses:
        diag_data = zscores[zscores["diagnosis"] == diag]

        fig = go.Figure()

        for region in region_names:
            region_metric_cols = [f"{region}__{metric}" for metric in metrics]

            # Skip missing regions
            if not all(col in diag_data.columns for col in region_metric_cols):
                continue

            median_abs_z = diag_data[region_metric_cols].abs().median()

            fig.add_trace(go.Scatterpolar(
                r=median_abs_z.values,
                theta=metrics,
                fill='toself',
                name=region,
                line=dict(color=region_colors[region])
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    title="|Median Z-score|",
                    range=[0, global_rmax]  # Normalized axis across all plots!
                )
            ),
            showlegend=True,
            title=f"Median Z-Score Deviation per Region ({diag})"
        )

        out_html = f"{output_prefix}_{diag}.html"
        fig.write_html(out_html)
        print(f"✅ Saved: {out_html}")

def plot_parallel_coordinates(csv_path, subject_id, out_html_prefix="parallel_plot"):
    df = pd.read_csv(csv_path)
    df = df[df["subject_id"] == subject_id]

    metrics = ["volume_mm3", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm",
               "surface_area_mm2", "compactness", "elongation"]

    regions = set(col.split("__")[0] for col in df.columns if "__" in col)

    data = []
    for _, row in df.iterrows():
        for region in regions:
            entry = {"age": row["age"], "region": region, "subject_id": row["subject_id"]}
            for metric in metrics:
                col = f"{region}__{metric}"
                if col in row:
                    entry[metric] = row[col]
            data.append(entry)

    long_df = pd.DataFrame(data)

    # Z-score normalization
    norm_df = long_df.copy()
    for metric in metrics:
        norm_df[metric] = (long_df[metric] - long_df[metric].mean()) / long_df[metric].std()

    # Create two parallel coordinates plots (raw and normalized)
    fig_raw = px.parallel_coordinates(
        long_df, dimensions=metrics, color="age",
        labels={m: m for m in metrics},
        color_continuous_scale="Viridis",
        title=f"Parallel Coordinates (Raw) - {subject_id}"
    )

    fig_norm = px.parallel_coordinates(
        norm_df, dimensions=metrics, color="age",
        labels={m: m for m in metrics},
        color_continuous_scale="Viridis",
        title=f"Parallel Coordinates (Z-score normalized) - {subject_id}"
    )

    # Save to HTML files instead of showing
    raw_html = f"{out_html_prefix}_raw_{subject_id}.html"
    norm_html = f"{out_html_prefix}_zscore_{subject_id}.html"
    fig_raw.write_html(raw_html)
    fig_norm.write_html(norm_html)

    print(f"✅ Saved raw plot to: {raw_html}")
    print(f"✅ Saved z-score normalized plot to: {norm_html}")

def plot_zscore_violin_zlocal(csv_zlocal_path, output_html="zlocal_violin_by_region.html"):
    # === Load Data ===
    df = pd.read_csv(csv_zlocal_path)

    # === Metrics ===
    metrics = [
        "volume_mm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "surface_area_mm2",
        "compactness",
        "elongation"
    ]

    # === Extract relevant columns
    metric_cols = [col for col in df.columns if any(col.endswith(f"{m}_zlocal") for m in metrics)]
    df_long = df.melt(
        id_vars=["subject_id", "diagnosis"],
        value_vars=metric_cols,
        var_name="region_metric",
        value_name="zscore"
    )
    df_long["region"] = df_long["region_metric"].apply(lambda x: x.split("__")[0])
    df_long["metric"] = df_long["region_metric"].apply(lambda x: x.split("__")[1].replace("_zlocal", ""))

    # === Define consistent diagnosis colors ===
    diagnosis_colors = {
        "CC": px.colors.qualitative.Set1[0],
        "mTBI": px.colors.qualitative.Set1[1],
        "OI": px.colors.qualitative.Set1[2]
    }

    # === Initialize figure for first metric ===
    first_metric = metrics[0]
    fig = go.Figure()

    for diag in df_long["diagnosis"].unique():
        subset = df_long[(df_long["metric"] == first_metric) & (df_long["diagnosis"] == diag)]
        fig.add_trace(go.Violin(
            x=subset["region"],
            y=subset["zscore"],
            name=diag,
            line_color=diagnosis_colors.get(diag, "gray"),
            hovertext=[
                f"Subject: {sid}<br>Diagnosis: {diag}<br>Z: {val:.2f}"
                for sid, val in zip(subset["subject_id"], subset["zscore"])
            ],
            hoverinfo="text",
            box_visible=True,
            meanline_visible=True
        ))

    # === Dropdown buttons for metric switching ===
    buttons = []
    for metric in metrics:
        y_data = [
            df_long[(df_long["metric"] == metric) & (df_long["diagnosis"] == diag)]["zscore"]
            for diag in df_long["diagnosis"].unique()
        ]
        buttons.append(dict(
            label=metric,
            method="update",
            args=[
                {"y": y_data},
                {"title": f"Z-scores per Region for {metric}"}
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            x=0.85,
            y=1.15
        )],
        title=f"Z-scores per Region for {first_metric}",
        yaxis_title="Z-score (local)",
        template="plotly_white",
        violingap=0.2,
        violinmode="group"
    )

    fig.write_html(output_html)
    print(f"✅ Violin plot saved to {output_html}")

def plot_zscore_violin_with_metric_selector(
    df_healthy_path, df_diagnoses_path, output_html="zscore_violin_by_region.html"
):
    # === Load Data ===
    df_healthy = pd.read_csv(df_healthy_path)
    df_diagnoses = pd.read_csv(df_diagnoses_path)

    # === Metrics ===
    metrics = [
        "volume_mm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "surface_area_mm2",
        "compactness",
        "elongation"
    ]

    _, zscores, metric_cols = calculate_z_scores(df_healthy, df_diagnoses, metrics)

    # === Melt data for plotting ===
    long_df = zscores.melt(
        id_vars=["diagnosis", "subject_id"],
        value_vars=metric_cols,
        var_name="region_metric",
        value_name="zscore"
    )
    long_df["region"] = long_df["region_metric"].apply(lambda x: x.split("__")[0])
    long_df["metric"] = long_df["region_metric"].apply(lambda x: x.split("__")[1])

    # === Print mean Z-scores per region and diagnosis ===
    print("Mean Z-scores per region and diagnosis:")
    mean_table = long_df.groupby(["diagnosis", "region"])["zscore"].mean().unstack()
    print(mean_table)

    # === Define consistent diagnosis colors ===
    diagnosis_colors = {
        "CC": px.colors.qualitative.Set1[0],
        "mTBI": px.colors.qualitative.Set1[1],
        "OI": px.colors.qualitative.Set1[2]
    }

    # === Initialize figure ===
    first_metric = metrics[0]
    fig = go.Figure()

    for diag in long_df["diagnosis"].unique():
        df_subset = long_df[(long_df["metric"] == first_metric) & (long_df["diagnosis"] == diag)]
        fig.add_trace(go.Violin(
            x=df_subset["region"],
            y=df_subset["zscore"],
            name=diag,
            line_color=diagnosis_colors.get(diag, "gray"),
            hovertext=[
                f"Subject: {sid}<br>Diagnosis: {diag}<br>Z: {val:.2f}"
                for sid, val in zip(df_subset["subject_id"], df_subset["zscore"])
            ],
            hoverinfo="text",
            box_visible=True,
            meanline_visible=True
        ))

    # === Dropdown for metric selection ===
    buttons = []
    for metric in metrics:
        visibility = []
        for diag in long_df["diagnosis"].unique():
            diag_mask = (long_df["metric"] == metric) & (long_df["diagnosis"] == diag)
            count = diag_mask.sum()
            visibility.extend([True if metric == first_metric else False] * (len(long_df["diagnosis"].unique())))
        buttons.append(dict(
            label=metric,
            method="update",
            args=[
                {"y": [
                    long_df[(long_df["metric"] == metric) & (long_df["diagnosis"] == diag)]["zscore"]
                    for diag in long_df["diagnosis"].unique()
                ]},
                {"title": f"Z-scores per Region for {metric}"}
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            x=0.85,
            y=1.15
        )],
        title=f"Z-scores per Region for {first_metric}",
        yaxis_title="Z-score",
        template="plotly_white",
        violingap=0.2,
        violinmode="group"
    )

    # Save to HTML
    fig.write_html(output_html)
    print(f"✅ Violin plot saved to {output_html}")

def plot_metric_trajectories_interactive(csv_path, out_html="trajectory_plot.html"):
    df = pd.read_csv(csv_path)

    # Filter to diagnosis and subjects with multiple sessions
    multi_session_subjects = df["subject_id"].value_counts()
    multi_session_subjects = multi_session_subjects[multi_session_subjects > 1].index
    df = df[df["subject_id"].isin(multi_session_subjects)]

    metrics = ["volume_mm3", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm",
               "surface_area_mm2", "compactness", "elongation"]

    regions = set(col.split("__")[0] for col in df.columns if "__" in col)

    # Normalize (z-score) for each metric across all data
    zscored_df = df.copy()
    for metric in metrics:
        metric_cols = [f"{r}__{metric}" for r in regions if f"{r}__{metric}" in df.columns]
        zscored_df[metric_cols] = (df[metric_cols] - df[metric_cols].mean()) / df[metric_cols].std()

    fig = go.Figure()

    for normalize, name in zip([False, True], ["Raw", "Z-score"]):
        sub_df = zscored_df if normalize else df

        for subject_id, group in sub_df.groupby("subject_id"):
            group = group.sort_values("age")
            values = []
            for region in regions:
                col = f"{region}__volume_mm3"
                if col in group.columns:
                    values.append(group[col].values)
            if values:
                mean_vals = pd.DataFrame(values).mean(axis=0)
                hover_text = [f"Subject: {subject_id}<br>Age: {age}" for age in group["age"]]

                fig.add_trace(go.Scatter(
                    x=group["age"],
                    y=mean_vals,
                    mode="lines+markers",
                    name=f"{subject_id} ({name})",
                    text=hover_text,
                    hoverinfo="text+y",
                    visible=normalize  # Start with normalized view only
                ))

    # Buttons for switching between raw and z-score views
    buttons = []
    for i, name in enumerate(["Raw", "Z-score"]):
        visibility = [v == (i == 1) for v in range(len(fig.data))]
        buttons.append(dict(
            label=name,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Trajectory of volume_mm3 ({name} values)"}]
        ))

    fig.update_layout(
        title="Trajectory of volume_mm3 (Z-score normalized view)",
        xaxis_title="Age",
        yaxis_title="Mean volume_mm3 across regions",
        template="plotly_white",
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            x=0.7,
            y=1.15
        )]
    )

    # Save the figure to HTML
    fig.write_html(out_html)
    print(f"✅ Saved interactive trajectory plot to: {out_html}")

def extract_region_features(segmentation, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    segmentation: 3D numpy array, where each voxel has a label according to SYNTHSEG_CODEMAP
    voxel_spacing: tuple, size of voxel (dx, dy, dz) in mm
    """
    results = {}

    voxel_volume = np.prod(voxel_spacing)  # mm³ per voxel

    for label_id, label_name in const.SYNTHSEG_CODEMAP.items():
        mask = (segmentation == label_id)

        if not np.any(mask):
            continue  # skip if region absent

        coords = np.argwhere(mask)  # (N, 3) array: (z, y, x)
        if coords.shape[0] < 5:
            print(f'Too small region: {label_name}, less than 5 coordinates; skipping calculations...')
            continue  # too small region, skip

        voxel_volume = np.prod(voxel_spacing)  # mm³
        voxel_area = np.mean(voxel_spacing) ** 2  # mm² for approximate surface area

        # === Volume
        volume = coords.shape[0] * voxel_volume

        # === Centroid
        centroid = coords.mean(axis=0) * np.array(voxel_spacing[::-1])  # (x, y, z)

        # === Surface Area (simple method)
        # surface_mask = measure.mesh_surface_area(mask.astype(float)) * np.mean(voxel_spacing)**2  # rough approx
        # Simpler surface area: count border voxels
        surface_area = np.sum(scipy.ndimage.binary_erosion(mask) != mask) * voxel_area

        # === Compactness
        compactness = (volume ** (2/3)) / (surface_area + 1e-8)

        # === Elongation
        pca = PCA(n_components=3)
        pca.fit(coords)
        elongation = pca.singular_values_[0] / (pca.singular_values_[2] + 1e-8)  # largest / smallest axis

        results[label_name] = {
            "volume_mm3": volume,
            "centroid_x_mm": centroid[2],
            "centroid_y_mm": centroid[1],
            "centroid_z_mm": centroid[0],
            "surface_area_mm2": surface_area,
            "compactness": compactness,
            "elongation": elongation,
        }

    return results

def extract_shape_features_to_csv(csv_path, output_features_csv, voxel_spacing=(1.0, 1.0, 1.0)):
    df = pd.read_csv(csv_path)
    all_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting shape features"):
        subject_id = row["subject_id"]
        image_uid = row["image_uid"]
        segm_path = row["segm_path"]
        age = row["age"]
        sex = row["sex"]

        if 'KOALA' in csv_path:
            diagnosis = row["diagnosis"]
            time_post_injury_days = row["time_post_injury_days"]

        if not isinstance(segm_path, str) or not os.path.exists(segm_path):
            print(f"❌ Missing segmentation for {subject_id} {image_uid}")
            continue

        try:
            segm_img = nib.load(segm_path)
            segm_data = segm_img.get_fdata()

            region_features = extract_region_features(segm_data, voxel_spacing=voxel_spacing)

            if 'KOALA' in csv_path:
                flat_record = {
                    "subject_id": subject_id,
                    "image_uid": image_uid,
                    "age": age,
                    "sex": sex,
                    "diagnosis": diagnosis,
                    "time_post_injury_days": time_post_injury_days
                }
            else:
                flat_record = {
                    "subject_id": subject_id,
                    "image_uid": image_uid,
                    "age": age,
                    "sex": sex
                }

            for region, features in region_features.items():
                for feat_name, value in features.items():
                    flat_record[f"{region}__{feat_name}"] = value

            all_records.append(flat_record)

        except Exception as e:
            print(f"⚠️ Failed for {subject_id} {image_uid}: {e}")

    features_df = pd.DataFrame(all_records)
    features_df.to_csv(output_features_csv, index=False)
    print(f"✅ Saved extracted features to {output_features_csv}")

def merge_csvs(csv1_path, csv2_path, output_path):
    # Load both CSVs
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Add missing columns explicitly
    if "diagnosis" not in df2.columns:
        df2["diagnosis"] = "CC"
    if "time_post_injury_days" not in df2.columns:
        df2["time_post_injury_days"] = pd.NA

    if "image_uid" not in df1.columns:
        df1["image_uid"] = pd.NA

    # Unify columns without sorting (preserve subject_id first)
    all_columns = list(df1.columns.union(df2.columns, sort=False))

    df1 = df1.reindex(columns=all_columns)
    df2 = df2.reindex(columns=all_columns)

    # Concatenate and save
    merged = pd.concat([df1, df2], ignore_index=True)

    # Ensure subject_id is first column
    cols = merged.columns.tolist()
    if "subject_id" in cols:
        cols = ["subject_id"] + [col for col in cols if col != "subject_id"]
        merged = merged[cols]

    merged.to_csv(output_path, index=False)
    print(f"✅ Merged CSV saved to: {output_path}")

if __name__ == "__main__":

    # extract_shape_features_to_csv(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/dataset.csv",
    # output_features_csv="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv",
    # voxel_spacing=(1.0, 1.0, 1.0)
    # )

    # extract_shape_features_to_csv(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/dataset_KOALA.csv",
    # output_features_csv="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features_KOALA.csv",
    # voxel_spacing=(1.0, 1.0, 1.0)
    # )

    # merge_csvs("/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features_KOALA.csv", 
    #            "/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv", 
    #            "/home/andim/projects/def-bedelb/andim/brlp-data/shape_features_all.csv")

    # plot_spider_charts_per_diagnosis(
    # healthy_csv="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv",
    # diagnoses_csv="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features_KOALA.csv"    
    # )

    # plot_parallel_coordinates(
    # csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features_KOALA.csv",
    # subject_id="sub-10098"
    # )

    # plot_zscore_violin_with_metric_selector(
    # df_healthy_path="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv",
    # df_diagnoses_path="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features_KOALA.csv",
    # output_html="violin_zscores_per_region_with_slider.html"
    # )

    # plot_metric_trajectories_interactive(
    #     csv_path="/home/andim/projects/def-bedelb/andim/brlp-data/healthy_shape_features.csv"
    # )

    # compute_and_plot_gams_from_shape_features(
    # csv_all="/home/andim/projects/def-bedelb/andim/brlp-data/shape_features_all.csv",
    # output_csv="/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all.csv",
    # model_dir="gam_models",
    # plot_html="gam_fits_age_months.html"
    # )

    ## Statistical testing
    compare_diagnosis_vs_control("/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all_local_1mth.csv", output_csv="group_diff_stats_1mth.csv")
    compare_diagnosis_vs_control("/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all_local.csv", output_csv="group_diff_stats_3mth.csv")

    compute_local_zscores_with_gam(
    csv_all="/home/andim/projects/def-bedelb/andim/brlp-data/shape_features_all.csv",
    output_csv="/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all_local_1-5mth.csv",
    model_dir="gam_models_local_1mth",
    window_months=1.5  # Use ±3 months around each age
    )

    compute_residual_std_comparison_per_metric(
    csv_all="/home/andim/projects/def-bedelb/andim/brlp-data/shape_features_all.csv",
    output_dir="resid_std_comparison_plots",
    window_months=1.5,
    step=1)

    plot_zscore_violin_zlocal('/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all_local_1-5mth.csv', output_html="zlocal_violin_by_region_1mth.html")

    # plot_spider_charts_zlocal("/home/andim/projects/def-bedelb/andim/brlp-data/zscored_all_local.csv")

    



    













