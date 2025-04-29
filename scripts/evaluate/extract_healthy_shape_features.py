import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import numpy as np
import scipy.ndimage
from skimage import measure
from sklearn.decomposition import PCA
from brlp import init_autoencoder, const

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
            continue  # too small region, skip

        # === Volume
        volume = coords.shape[0] * voxel_volume  # in mm³

        # === Centroid
        centroid = coords.mean(axis=0) * np.array(voxel_spacing[::-1])  # (x, y, z)

        # === Surface Area
        surface_mask = measure.mesh_surface_area(mask.astype(float)) * np.mean(voxel_spacing)**2  # rough approx
        # If you want a simpler surface area: count border voxels
        # surface_area = np.sum(scipy.ndimage.binary_erosion(mask) != mask) * voxel_area

        # === Compactness
        compactness = (volume ** (2/3)) / (surface_mask + 1e-8)

        # === Elongation
        pca = PCA(n_components=3)
        pca.fit(coords)
        elongation = pca.singular_values_[0] / (pca.singular_values_[2] + 1e-8)  # largest / smallest axis

        results[label_name] = {
            "volume_mm3": volume,
            "centroid_x_mm": centroid[2],
            "centroid_y_mm": centroid[1],
            "centroid_z_mm": centroid[0],
            "surface_area_mm2": surface_mask,
            "compactness": compactness,
            "elongation": elongation,
        }

    return results

# === CONFIG ===
csv_path = "/path/to/A.csv"
output_features_csv = "healthy_shape_features.csv"
voxel_spacing = (1.0, 1.0, 1.0)  # modify if needed

# === Load Metadata ===
df = pd.read_csv(csv_path)
healthy_df = df[df.split == "train"]  # or however you define healthy split

# === Initialize all extracted records ===
all_records = []

for _, row in tqdm(healthy_df.iterrows(), total=len(healthy_df), desc="Extracting shape features"):
    subject_id = row["subject_id"]
    image_uid = row["image_uid"]
    segm_path = row["segm_path"]

    if not isinstance(segm_path, str) or not os.path.exists(segm_path):
        print(f"❌ Missing segmentation for {subject_id} {image_uid}")
        continue

    try:
        segm_img = nib.load(segm_path)
        segm_data = segm_img.get_fdata()

        region_features = extract_region_features(segm_data, voxel_spacing=voxel_spacing)

        # Flatten the dict to a single row
        flat_record = {"subject_id": subject_id, "image_uid": image_uid}
        for region, features in region_features.items():
            for feat_name, value in features.items():
                flat_record[f"{region}__{feat_name}"] = value

        all_records.append(flat_record)

    except Exception as e:
        print(f"⚠️ Failed for {subject_id} {image_uid}: {e}")

# === Save to CSV ===
features_df = pd.DataFrame(all_records)
features_df.to_csv(output_features_csv, index=False)

print(f"✅ Saved extracted features to {output_features_csv}")
