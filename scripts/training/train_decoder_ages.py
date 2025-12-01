import os
import argparse
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import L1Loss
import copy
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from PIL import Image
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from monai.metrics.regression import compute_ssim_and_cs, KernelType
from monai.utils import set_determinism
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    SpacingD,
    ResizeWithPadOrCropD,
    ScaleIntensityD,
    Compose
)
from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator
)

def batch_ssim_scalar(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    ssim_map, _ = compute_ssim_and_cs(
        y_pred=y_pred, y=y_true,
        spatial_dims=3,
        kernel_size=(11, 11, 11),
        kernel_sigma=(1.5, 1.5, 1.5),
        data_range=data_range,
        kernel_type=KernelType.GAUSSIAN,
        k1=0.01, k2=0.03,
    )
    B = ssim_map.shape[0]
    return ssim_map.view(B, -1).mean(dim=1).mean()  # scalar

def log_recon_3x4_to_wandb(
    meta: dict,
    t1_vol_np: np.ndarray,
    t1_recon_np: np.ndarray,
    t2_true_np: np.ndarray,
    t2_pred_np: np.ndarray,
    step: int = None,
    tag: str = "Recon 3x4"
):
    """
    Logs a single 3×4 grid for:
      [T1 input, T1 VAE‐recon, T2 true, T2 predicted]
    across [axial, coronal, sagittal] views.

    Args:
        meta (dict): must contain at least:
            'subject_id', 'start_age', 'followup_age', optional 'image_uid', error metrics...
        t1_vol_np      ([1,D,H,W]): baseline input
        t1_recon_np    ([1,D,H,W]): VAE‐reconstruction of baseline
        t2_true_np     ([1,D,H,W]): true follow-up
        t2_pred_np     ([1,D,H,W]): predicted follow-up
        step (int): W&B step
        tag (str): W&B key
    """
    # Drop channel dim, get [D,H,W]
    t1  = t1_vol_np[0]
    r1  = t1_recon_np[0]
    t2  = t2_true_np[0]
    p2  = t2_pred_np[0]
    D,H,W = t1.shape

    # mid‐slice indices
    iz = D//2
    iy = H//2
    ix = W//2

    # prepare 12 images in row‐major order:
    images = [
        # Axial row (slice along axis=0)
        t1 [iz,:,:], r1 [iz,:,:], t2 [iz,:,:], p2 [iz,:,:],
        # Coronal row (axis=1)
        t1 [:,iy,:], r1 [:,iy,:], t2 [:,iy,:], p2 [:,iy,:],
        # Sagittal row (axis=2)
        t1 [:,:,ix], r1 [:,:,ix], t2 [:,:,ix], p2 [:,:,ix],
    ]

    # titles for each subplot
    sa = meta.get('start_age', None)
    fa = meta.get('followup_age', None)
    sid = meta.get('subject_id','')
    uid = meta.get('image_uid','')
    top_title = f"{sid}  |  ΔAge={(fa-sa):.2f} yr" if sa is not None and fa is not None else sid
    if uid: top_title += f"  |  UID={uid}"
    cols = ["T₁ Input", "T₁ Recon", "T₂ True", "T₂ Pred"]
    rows = ["Axial", "Coronal", "Sagittal"]
    titles = []
    for view in rows:
        for col in cols:
            if view=="Axial" and col=="T₁ Input" and sa is not None:
                titles.append(f"{col}\n(age={sa:.2f})")
            elif view=="Axial" and col=="T₂ True" and fa is not None:
                titles.append(f"{col}\n(age={fa:.2f})")
            else:
                titles.append(col)

    # plot 3×4
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    fig.suptitle(top_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save, log, remove
    tmp = f"{args.output_dir}/tmp_{sid}_{uid}_3x4.png"
    plt.savefig(tmp, dpi=120)
    plt.close(fig)
    if step is not None:
        wandb.log({tag: wandb.Image(tmp)}, step=step)
    else:
        wandb.log({tag: wandb.Image(tmp)})
    os.remove(tmp)

def log_recon_comparison_to_wandb(
    meta: dict,
    baseline_vol_np: np.ndarray,
    true_vol_np: np.ndarray,
    pred_vol_np: np.ndarray,
    step: int = None,
    tag: str = "Reconstruction Comparison"
):
    """
    Builds a 3×3 grid showing the middle-axial, coronal, and sagittal slices of:
      [baseline_input, true_followup, predicted_followup]
    and logs that single figure to W&B under `tag` and `step`.

    Args:
        meta (dict): Dictionary containing at least:
                     - 'subject_id'    (str or int)
                     - 'start_age'     (float, in years)
                     - 'followup_age'  (float, in years)
                     - (optional) 'l2_error', 'mse', 'error_diff'
                     - (optional) 'image_uid'
        baseline_vol_np (np.ndarray): Baseline volume, shape [1, D, H, W]
        true_vol_np     (np.ndarray): True follow-up volume, shape [1, D, H, W]
        pred_vol_np     (np.ndarray): Predicted follow-up volume, shape [1, D, H, W]
        step (int, optional): W&B step index. If None, step is omitted.
        tag (str): The W&B key under which to log this image.
    """
    # Extract spatial dims from shape [1, D, H, W]
    baseline = baseline_vol_np[0]  # [D, H, W]
    truef    = true_vol_np[0]
    predf    = pred_vol_np[0]

    D, H, W = baseline.shape

    # Compute middle indices
    mid_axial = D // 2
    mid_cor   = H // 2
    mid_sag   = W // 2

    # --- Axial slices (slice along axis=0) ---
    base_ax = baseline[mid_axial, :, :]
    true_ax = truef[mid_axial,    :, :]
    pred_ax = predf[mid_axial,    :, :]

    # --- Coronal slices (slice along axis=1) ---
    base_cor = baseline[:, mid_cor, :]
    true_cor = truef[:,    mid_cor, :]
    pred_cor = predf[:,    mid_cor, :]

    # --- Sagittal slices (slice along axis=2) ---
    base_sag = baseline[:, :, mid_sag]
    true_sag = truef[:,    :, mid_sag]
    pred_sag = predf[:,    :, mid_sag]

    # Build subplot titles
    subj_id      = meta.get('subject_id', '')
    start_age    = meta.get('start_age', None)
    followup_age = meta.get('followup_age', None)
    l2_err       = meta.get('l2_error', None)
    mse          = meta.get('mse', None)
    err_diff     = meta.get('error_diff', None)
    image_uid    = meta.get('image_uid', '')

    top_title = f"{subj_id}"
    if (start_age is not None) and (followup_age is not None):
        top_title += f"  |  ΔAge = {followup_age - start_age:.2f} yr"
    if (l2_err is not None) and (mse is not None) and (err_diff is not None):
        top_title += f"  |  L2={l2_err:.2f}  MSE={mse:.2f}  ΔErr={err_diff:.2f}"
    if image_uid:
        top_title += f"  |  UID={image_uid}"

    titles = [
        f"T₁ Baseline Axial\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Axial",
        f"T₂ True Axial\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Axial",
        "T₂ Pred Axial",
        f"T₁ Baseline Coronal\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Coronal",
        f"T₂ True Coronal\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Coronal",
        "T₂ Pred Coronal",
        f"T₁ Baseline Sagittal\n(age={start_age:.2f})" if start_age is not None else "T₁ Baseline Sagittal",
        f"T₂ True Sagittal\n(age={followup_age:.2f})" if followup_age is not None else "T₂ True Sagittal",
        "T₂ Pred Sagittal"
    ]

    images = [
        base_ax,   true_ax,   pred_ax,
        base_cor,  true_cor,  pred_cor,
        base_sag,  true_sag,  pred_sag
    ]

    # Create the 3×3 figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Put a global suptitle
    fig.suptitle(top_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to a temporary file, log it, then delete
    tmp_path = f"tmp_{subj_id}_{image_uid}_recon.png"
    plt.savefig(tmp_path, dpi=120)
    plt.close(fig)

    if step is not None:
        wandb.log({tag: wandb.Image(tmp_path)}, step=step)
    else:
        wandb.log({tag: wandb.Image(tmp_path)})

    os.remove(tmp_path)

def log_differences_in_predictions(autoencoder, img0, img1, total_counter):
    # 1) Compute recon_t0, recon_t1
    with torch.no_grad():
        recon0 = autoencoder.reconstruct(img0[:1])
        recon1 = autoencoder.reconstruct(img1[:1])
    recon_t0 = recon0[0,0].cpu().numpy()[None]
    recon_t1 = recon1[0,0].cpu().numpy()[None]

    # 2) Compute diffs and log them
    # D1 = true_vol[0] - baseline_vol[0] # T2true - T1
    # D2 = true_vol[0] - recon_t0[0] #T2true - T1recon
    D1 = np.abs(pred_vol[0] - true_vol[0]) #T2pred-T2true
    D2 = np.abs(pred_vol[0] - baseline_vol[0]) #T2pred-T1
    D3 = np.abs(pred_vol[0] - recon_t0[0]) #T2pred-T1recon
    D4 = np.abs(true_vol[0] - baseline_vol[0]) #T2true-T1
    D5 = np.abs(recon_t1[0] - recon_t0[0]) #T2recon-T1recon
    m = max(D.max() for D in (D1,D2,D3,D4,D5))
    iz = D1.shape[0]//2
    # make a 1×5 figure
    fig, axes = plt.subplots(1, 5, figsize=(15,4), constrained_layout=True)
    titles = ["|T2pred–T2true|","|T2pred–T1|","|T2pred–T1recon|","|T2true–T1|","|T2recon–T1recon|"]
    for ax, D, ttl in zip(axes, (D1,D2,D3,D4,D5), titles):
        im = ax.imshow(np.rot90(D[iz]), cmap="hot", vmin=0, vmax=m)
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    # add single colorbar on the right
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Abs. Error", rotation=90)

    delta_age = age1_months - age0_months
    fig.suptitle(f"Epoch {epoch} | {subject_id[0]}  |  ΔAge={delta_age:.2f} yr", fontsize=12)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save + log
    tmpd = f"{args.output_dir}/abs_errors_e{epoch}_it{total_counter}.png"
    fig.savefig(tmpd, dpi=100)
    plt.close(fig)

    # log to W&B
    wandb.log({ "Errors/5-way": wandb.Image(tmpd) }, step=total_counter)

    # 3) If you want mean‐absolute‐errors of any pair:
    mae_pred_true   = float(D1.mean())   # for T2pred–T2true
    mae_pred_t1     = float(D2.mean())
    mae_pred_t1rec  = float(D3.mean())
    mae_true_t1     = float(D4.mean())
    mae_recon_t1rec = float(D5.mean())

    wandb.log({
        "mae/T2pred-T2true":      mae_pred_true,
        "mae/T2pred-T1":        mae_pred_t1,
        "mae/T2pred-T1recon":   mae_pred_t1rec,
        "mae/T2true-T1":        mae_true_t1,
        "mae/T2recon-T1recon":         mae_recon_t1rec,
        "delta_age":          delta_age
    }, step=total_counter)

    return tmpd

# ── Define AgeEmbedMLP and AgeConditionalDecoder exactly as before ──

# class AgeEmbedMLP(nn.Module):
#     """
#     Simple two‐layer MLP: takes scalar age ∈ [0,1] → age_embed_dim vector.
#     """
#     def __init__(self, age_embed_dim=16):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, age_embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(age_embed_dim, age_embed_dim),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, age_scalar):
#         # age_scalar: [B,1]
#         return self.mlp(age_scalar)    # → [B, age_embed_dim]
    
class AgeEmbedMLP(nn.Module):
    def __init__(self, in_dim=3, age_embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, age_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, age_vec):  # [B, in_dim]
        return self.mlp(age_vec)
class AgeConditionalDecoder(nn.Module):
    """
    Takes:
      - z: [B, latent_channels, D_lat, H_lat, W_lat]
      - age_scalar: [B, 1] normalized ∈ [0,1]
    and outputs:
      - x_pred: [B, 1, D_img, H_img, W_img]
    The workflow:
      1. Flatten z_post via post_quant_conv into [B, latent_flat]
      2. Concatenate with age_embed(age_scalar) → [B, latent_flat + age_embed_dim]
      3. Reproject to [B, latent_flat] via a linear (cond_proj), reshape to [B, C_lat, D_lat, H_lat, W_lat]
      4. Run through original decoder blocks → [B, 1, D_img, H_img, W_img]
    """
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=16):
        super().__init__()
        self.age_embed_dim = age_embed_dim

        # Copy post_quant_conv and decoder blocks from the pretrained AE
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_blocks = copy.deepcopy(pretrained_ae.decoder.blocks)  # ModuleList

        # Age MLP
        self.age_mlp = AgeEmbedMLP(age_embed_dim=age_embed_dim)

        # cond_proj to be built once we know z’s shape
        self.cond_proj = None
        self._proj_built = False

        # Wrap decoder blocks into an nn.Sequential
        self.decoder_body = nn.Sequential(*self.decoder_blocks)

    def initialize_projection(self, z_shape, device):
        """
        Build cond_proj when we know z_shape = (B, C_lat, D_lat, H_lat, W_lat).
        """
        B, C_lat, D_lat, H_lat, W_lat = z_shape
        latent_flat = C_lat * D_lat * H_lat * W_lat
        self.latent_flat = latent_flat
        # New linear: (latent_flat + age_embed_dim) → latent_flat
        self.cond_proj = nn.Linear(latent_flat + self.age_embed_dim, latent_flat).to(device)
        nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity='relu')
        if self.cond_proj.bias is not None:
            nn.init.constant_(self.cond_proj.bias, 0.0)
        self._proj_built = True

    def forward(self, z, age_scalar):
        """
        z: [B, C_lat, D_lat, H_lat, W_lat]
        age_scalar: [B,1]
        """
        B, C_lat, D_lat, H_lat, W_lat = z.shape

        # Build cond_proj on first forward
        if not self._proj_built:
            self.initialize_projection(z.shape, z.device)

        # 1) Post‐quant convolution
        z_post = self.post_quant_conv(z)  # [B, C_lat, D_lat, H_lat, W_lat]

        # 2) Flatten to [B, latent_flat]
        z_flat = z_post.view(B, -1)

        # 3) Embed age
        age_vec = self.age_mlp(age_scalar)  # [B, age_embed_dim]

        # 4) Concatenate
        combined = torch.cat([z_flat, age_vec], dim=1)  # [B, latent_flat + age_embed_dim]

        # 5) Project back to latent_flat
        z_proj_flat = self.cond_proj(combined)  # [B, latent_flat]

        # 6) Reshape to [B, C_lat, D_lat, H_lat, W_lat]
        z_proj = z_proj_flat.view(B, C_lat, D_lat, H_lat, W_lat)

        # 7) Run through original decoder cascade
        x_pred = self.decoder_body(z_proj)  # [B, 1, D_img, H_img, W_img]
        return x_pred

class AgeVectorConditionalDecoder(nn.Module):
    def __init__(self, pretrained_ae: torch.nn.Module, age_embed_dim=16):
        super().__init__()
        self.age_embed_dim = age_embed_dim
        self.post_quant_conv = copy.deepcopy(pretrained_ae.post_quant_conv)
        self.decoder_blocks  = copy.deepcopy(pretrained_ae.decoder.blocks)
        self.decoder_body    = nn.Sequential(*self.decoder_blocks)
        self.age_mlp         = AgeEmbedMLP(in_dim=3, age_embed_dim=age_embed_dim)
        self.cond_proj = None
        self._proj_built = False

    def initialize_projection(self, z_shape, device):
        B, C_lat, D_lat, H_lat, W_lat = z_shape
        latent_flat = C_lat * D_lat * H_lat * W_lat
        self.latent_flat = latent_flat
        self.cond_proj = nn.Linear(latent_flat + self.age_embed_dim, latent_flat).to(device)
        nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity='relu')
        if self.cond_proj.bias is not None:
            nn.init.constant_(self.cond_proj.bias, 0.0)
        self._proj_built = True

    def forward(self, z, age0, age1):
        if not self._proj_built:
            self.initialize_projection(z.shape, z.device)

        # ensure same device/dtype
        age0 = age0.to(z.device).float()
        age1 = age1.to(z.device).float()

        z_post = self.post_quant_conv(z)
        B = z_post.size(0)
        z_flat = z_post.view(B, -1)

        d_age = age1 - age0
        # concat ages to feed the MLP: [age0, age1, d_age], all shape [B,1]
        age_vec = torch.cat([age0, age1, d_age], dim=1)   # [B,3]
        age_emb = self.age_mlp(age_vec)                   # [B, age_embed_dim]

        combined     = torch.cat([z_flat, age_emb], dim=1)
        z_proj_flat  = self.cond_proj(combined)
        C_lat, D_lat, H_lat, W_lat = z_post.shape[1:]
        z_proj = z_proj_flat.view(B, C_lat, D_lat, H_lat, W_lat)
        x_pred = self.decoder_body(z_proj)
        return x_pred

# ─── Define FullPredictor (encoder frozen + cond_decoder) ───────
class FullPredictor(nn.Module):
    def __init__(self, encoder, cond_decoder):
        super().__init__()
        self.encoder = encoder
        self.cond_decoder = cond_decoder

    def forward(self, x0, age0, age1):
        with torch.no_grad():
            mu, sigma = self.encoder.encode(x0)
            z = self.encoder.sampling(mu, sigma)  # (tip: use mu for less noise)
        return self.cond_decoder(z, age0, age1)
# class FullPredictor(nn.Module):
#     def __init__(self, encoder, cond_decoder):
#         super().__init__()
#         self.encoder = encoder       # frozen
#         self.cond_decoder = cond_decoder

#     def forward(self, x0, age_target):
#         # x0: [B,1,D,H,W], age_target: [B,1]
#         with torch.no_grad():
#             z_mu, z_sigma = self.encoder.encode(x0)
#             z = self.encoder.sampling(z_mu, z_sigma)  # [B, latent_channels, D_lat, H_lat, W_lat]
#         x1_pred = self.cond_decoder(z, age_target)    # [B,1,D,H,W]
#         return x1_pred

class LongitudinalMRIDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - subject_id
      - split
      - starting_image_path
      - followup_image_path
      - starting_age
      - followup_age
      - starting_image_uid
      - followup_image_uid
      (…plus whatever other columns you have)
    """

    def __init__(self, df: pd.DataFrame, resolution, target_shape):
        super().__init__()
        required = {
            'subject_id',
            'starting_image_path', 'followup_image_path',
            'starting_age',        'followup_age',
            'starting_image_uid',  'followup_image_uid'
        }
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.transforms = Compose([
            LoadImageD(keys=['img'], image_only=True),
            EnsureChannelFirstD(keys=['img']),
            SpacingD(pixdim=resolution, keys=['img']),
            ResizeWithPadOrCropD(spatial_size=target_shape, mode='minimum', keys=['img']),
            ScaleIntensityD(minv=0.0, maxv=1.0, keys=['img']),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Load volume‐0 (baseline) and volume‐1 (follow‐up)
        baseline_t = self.transforms({'img': row['starting_image_path']})['img']
        followup_t = self.transforms({'img': row['followup_image_path']})['img']
        img0 = baseline_t.float()  # [1, D, H, W]
        img1 = followup_t.float()  # [1, D, H, W]

        # 2) Normalize ages [1..7] → [0..1]
        # age0 = (row['starting_age'] * 6 - 1.0) / 6.0
        # age1 = (row['followup_age'] * 6  - 1.0) / 6.0
        age0 = (row['starting_age']) 
        age1 = (row['followup_age'])
        age0 = torch.tensor([age0], dtype=torch.float32)
        age1 = torch.tensor([age1], dtype=torch.float32)

        # 3) Grab subject_id and followup_image_uid
        subject_id     = row['subject_id']            # e.g. “sub-10098”
        # image_uid      = row['followup_image_uid']    # e.g. “ses-002”

        return img0, age0, img1, age1, subject_id

def _is_better(curr, best, delta_abs=0.0, delta_rel=None):
    if best is None:
        return True
    if delta_rel is not None:
        return curr < best * (1.0 - float(delta_rel))  # relative improvement
    return curr < (best - float(delta_abs))            # absolute improvement

def _save_ckpt(path, cond_decoder, discriminator, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "cond_decoder": cond_decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
        "extra": extra or {}
    }
    torch.save(state, path)

# @torch.no_grad()
# def evaluate_one_epoch(model, discriminator, loader, device, l1_loss_fn, perc_loss_fn,
#                        adv_loss_fn, adv_weight, perceptual_weight):
#     model.eval()
#     disc_eval_mode = True
#     if hasattr(discriminator, "training"):
#         discriminator.eval()

#     n = 0
#     sums = {
#         "rec": 0.0, "adv": 0.0, "perc": 0.0, "total": 0.0, "d": 0.0
#     }

#     for batch in loader:
#         img0, age0, img1, age1, subject_id = batch
#         img0 = img0.to(device); img1 = img1.to(device)
#         age0 = age0.to(device).float(); age1 = age1.to(device).float()

#         with autocast(enabled=True):
#             x1_pred = model(img0, age0, age1)

#             # generator-side components (no optimizer step)
#             logits_fake = discriminator(x1_pred.contiguous().float())[-1]
#             gen_adv_loss = adv_weight * adv_loss_fn(
#                 logits_fake, target_is_real=True, for_discriminator=False
#             )
#             rec_loss  = l1_loss_fn(x1_pred.float(), img1.float())
#             perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())

#             loss_g = rec_loss + gen_adv_loss + perc_loss

#             # discriminator eval loss (optional)
#             logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
#             d_loss_fake = adv_loss_fn(
#                 logits_fake_detach, target_is_real=False, for_discriminator=True
#             )
#             logits_real = discriminator(img1.contiguous().detach())[-1]
#             d_loss_real = adv_loss_fn(
#                 logits_real, target_is_real=True, for_discriminator=True
#             )
#             loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

#         bsz = img0.size(0)
#         n += bsz
#         sums["rec"]    += bsz * rec_loss.item()
#         sums["adv"]    += bsz * gen_adv_loss.item()
#         sums["perc"]   += bsz * perc_loss.item()
#         sums["total"]  += bsz * loss_g.item()
#         sums["d"]      += bsz * loss_d.item()

#     means = {k: v / max(n, 1) for k, v in sums.items()}
#     return means

@torch.no_grad()
def evaluate_one_epoch(model, discriminator, loader, device, l1_loss_fn, perc_loss_fn,
                       adv_loss_fn, adv_weight, perceptual_weight,
                       autoencoder=None):

    model.eval()
    if hasattr(discriminator, "training"):
        discriminator.eval()
    if autoencoder is not None:
        autoencoder.eval()

    n = 0
    sums = {
        "rec": 0.0, "adv": 0.0, "perc": 0.0, "total": 0.0, "d": 0.0,
        "ssim_pred_t2": 0.0, "ssim_t1_t2": 0.0, "ssim_pred_t1recon": 0.0,
    }

    for batch in loader:
        img0, age0, img1, age1, subject_id = batch
        img0 = img0.to(device); img1 = img1.to(device)
        age0 = age0.to(device).float(); age1 = age1.to(device).float()

        with autocast(enabled=True):
            x1_pred = model(img0, age0, age1)

            # generator-side components
            logits_fake = discriminator(x1_pred.contiguous().float())[-1]
            gen_adv_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
            rec_loss  = l1_loss_fn(x1_pred.float(), img1.float())
            perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())
            loss_g = rec_loss + gen_adv_loss + perc_loss

            # discriminator eval loss (optional)
            logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
            d_loss_fake = adv_loss_fn(logits_fake_detach, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(img1.contiguous().detach())[-1]
            d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

            # ---- SSIMs ----
            ssim_pred_t2 = batch_ssim_scalar(x1_pred.float(), img1.float(), data_range=1.0)
            ssim_t1_t2   = batch_ssim_scalar(img0.float(),   img1.float(), data_range=1.0)

            if autoencoder is not None:
                t1_recon = autoencoder.reconstruct(img0)   # adjust if your API differs
                ssim_pred_t1recon = batch_ssim_scalar(x1_pred.float(), t1_recon.float(), data_range=1.0)
            else:
                ssim_pred_t1recon = torch.tensor(float("nan"), device=device)

        bsz = img0.size(0)
        n += bsz
        sums["rec"]   += bsz * rec_loss.item()
        sums["adv"]   += bsz * gen_adv_loss.item()
        sums["perc"]  += bsz * perc_loss.item()
        sums["total"] += bsz * loss_g.item()
        sums["d"]     += bsz * loss_d.item()

        sums["ssim_pred_t2"]     += bsz * float(ssim_pred_t2.item())
        sums["ssim_t1_t2"]       += bsz * float(ssim_t1_t2.item())
        sums["ssim_pred_t1recon"]+= bsz * float(ssim_pred_t1recon.item()) if np.isfinite(ssim_pred_t1recon.item()) else 0.0

    return {k: v / max(n, 1) for k, v in sums.items()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str,
                        help="CSV with columns: split, starting_image_path, followup_image_path, starting_age, followup_age")
    parser.add_argument('--cache_dir',      default=None, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--adv_weight',         default=0.025, type=float)
    parser.add_argument('--perceptual_weight',  default=0.001, type=float)
    parser.add_argument('--discriminative_lr',  default=0.1, type=float,
                        help="How much to adapt old layers versus new ones for fine-tuning (At 0.1; old layers do adapt, but at a slower pace than new ones)")
    parser.add_argument('--age_embed_dim',  default=16, type=int)
    parser.add_argument('--decoder_k_blocks_to_unfreeze',  default=2, type=int)
    parser.add_argument('--project',        default="age_conditional_decoder_ages", type=str,
                        help="W&B project name")
    parser.add_argument('--run_name',       default=None,   type=str,
                        help="W&B run name (if omitted, it puts the date and time the experiment was run)")
    parser.add_argument('--fold',      required=True,  type=int,
                        help="Which current training fold (1-5)")
    # New arguments for final run
    parser.add_argument('--early_stop_patience', type=int, default=50,
                    help='Stop if no val improvement for this many epochs.')
    parser.add_argument('--early_stop_delta', type=float, default=0.0,
                        help='Minimum improvement in val/Generator/total to count as better.')
    parser.add_argument('--min_epochs', type=int, default=150,
                        help='Always run at least this many epochs before early stopping.')
    # LR plateau scheduler knobs
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_patience', type=int, default=12)
    parser.add_argument('--plateau_cooldown', type=int, default=6)
    parser.add_argument('--plateau_max_reductions', type=int, default=3)
    args = parser.parse_args()

    set_determinism(0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fold = args.fold

    # To cumulate difference plots and then create a gif
    diff_frames = []

    # Initialize WandB
    wandb.init(
        project=args.project,
        name=f"{run_name}_{fold}",
        mode="online",
        config=vars(args)
    )
    config = wandb.config

    wandb.run.define_metric("train/*", step_metric="total_counter")
    wandb.run.define_metric("val/*",   step_metric="epoch")

    # ─── Build train/val splits ────────────────────────────────────
    dataset_df = pd.read_csv(config.dataset_csv)
    
    if fold not in range(1,6):
        raise ValueError(f"fold must be 1..5, got {fold}")

    print(f"\n\n=== Training on fold {fold} ===")
    fold_col = 'fold'
    split_col = 'split'
    # keep only this fold
    df_fold = dataset_df[dataset_df[fold_col] == fold].reset_index(drop=True)

    # train/val for training; test kept for later heavy evaluation
    train_df = df_fold[df_fold[split_col] == "train"].reset_index(drop=True)
    val_df   = df_fold[df_fold[split_col] == "val"].reset_index(drop=True)
    # test_df  = df_fold[df_fold[split_col] == "test"].reset_index(drop=True)

    print(f"[fold {fold}] pairs: train={len(train_df)} val={len(val_df)}")
    assert len(train_df) > 0 and len(val_df) > 0, "Empty train/val split for this fold!"

    # ─── Datasets & DataLoaders ────────────────────────────────────
    train_ds = LongitudinalMRIDataset(df=train_df,
                                    resolution=const.RESOLUTION,
                                    target_shape=const.INPUT_SHAPE_AE)
    
    val_ds = LongitudinalMRIDataset(df=val_df,
                                    resolution=const.RESOLUTION,
                                    target_shape=const.INPUT_SHAPE_AE)
    
    train_loader = DataLoader(dataset=train_ds,
                            batch_size=args.max_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    valid_loader = DataLoader(dataset=val_ds, 
                              num_workers=args.num_workers, 
                              batch_size=args.max_batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)
    
    print(f"[debug] train={len(train_ds)}  val={len(val_ds)}")
    assert len(val_ds) > 0, f"Fold {fold} has no validation rows."

    # ─── Load pretrained AutoencoderKL and Discriminator ────────────
    autoencoder = init_autoencoder(config.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(config.disc_ckpt).to(DEVICE)

    # ─── Freeze encoder & most of decoder ──────────────────────────
    for p in autoencoder.encoder.parameters():
        p.requires_grad = False
    for p in autoencoder.post_quant_conv.parameters():
        p.requires_grad = False
    for p in autoencoder.decoder.parameters():
        p.requires_grad = False

    # ─── Instantiate AgeConditionalDecoder ────────────────────────
    age_embed_dim = args.age_embed_dim
    cond_decoder = AgeVectorConditionalDecoder(pretrained_ae=autoencoder, age_embed_dim=age_embed_dim).to(DEVICE)

    # ─── PRE‐BUILD cond_proj by running a dummy forward pass ─────────
    with torch.no_grad():
        # Make a zero‐tensor of shape [1, 1, D, H, W] to match INPUT_SHAPE_AE
        dummy_img = torch.zeros(1, 1, *const.INPUT_SHAPE_AE, device=DEVICE)
        z_mu, z_sigma = autoencoder.encode(dummy_img)
        z_dummy = autoencoder.sampling(z_mu, z_sigma)
    # Now z_dummy.shape == (1, latent_channels, D_lat, H_lat, W_lat)
    cond_decoder.initialize_projection(z_dummy.shape, DEVICE)
    # ─────────────────────────────────────────────────────────────────
    # Unfreeze last k blocks (e.g. k=2)
    k = args.decoder_k_blocks_to_unfreeze
    layers = list(cond_decoder.decoder_body.children())
    for layer in layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True

    model = FullPredictor(autoencoder, cond_decoder).to(DEVICE)

    # ─── Losses & Optimizers ─────────────────────────────────────
    l1_loss_fn = L1Loss().to(DEVICE)

    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2
        ).to(DEVICE)

    # Only train: cond_decoder.age_mlp & cond_decoder.cond_proj (and any unfrozen decoder blocks)
    params_new = list(cond_decoder.age_mlp.parameters()) + list(cond_decoder.cond_proj.parameters())
    params_old = [p for p in cond_decoder.decoder_body.parameters() if p.requires_grad]

    optimizer_g = torch.optim.Adam([
        {'params': params_new, 'lr': config.lr},
        {'params': params_old, 'lr': config.lr * args.discriminative_lr}
    ], weight_decay=1e-5)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr)

    # Gradient accumulation (unchanged)
    gradacc_g = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )
    gradacc_d = GradientAccumulation(
        actual_batch_size=args.max_batch_size,
        expect_batch_size=args.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler()
    )

    avgloss = utils.AverageLoss()
    total_counter = 0
    best_val = None
    best_epoch = -1
    epochs_no_improve = 0

    best_ckpt_path = os.path.join(args.output_dir, 'best.ckpt')   # single rolling “best”
    last_ckpt_path = os.path.join(args.output_dir, 'last.ckpt')   # last epoch

    # Smoothing buffers
    val_hist = deque(maxlen=3)   # rolling median-of-3
    val_ema  = None              # EMA(alpha=0.3)

    # LR-on-plateau scheduler (on generator; add one for D if you want)
    sched_g = ReduceLROnPlateau(
        optimizer_g, mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        cooldown=args.plateau_cooldown,
        threshold=1e-4, threshold_mode='abs'
    )
    num_lr_reductions = 0
    prev_lr = optimizer_g.param_groups[0]['lr']


    # ─── Training Loop ───────────────────────────────────────────
    for epoch in range(config.n_epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            # Unpack the 5 items from LongitudinalMRIDataset.__getitem__
            img0, age0, img1, age1, subject_id = batch
            img0 = img0.to(DEVICE)   # [B,1,D,H,W]
            img1 = img1.to(DEVICE)
            age1 = age1.to(DEVICE).float()   # use follow-up age
            age0 = age0.to(DEVICE).float()

            # ── Generator (“conditional decoder”) step ─────────────
            with autocast(enabled=True):
                x1_pred = model(img0, age0, age1)  # [B,1,D,H,W]

                # Discriminator on fake → generator adv‐loss
                logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                gen_adv_loss = args.adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False
                )

                # Reconstruction (L1) between x1_pred and img1
                rec_loss = l1_loss_fn(x1_pred.float(), img1.float())

                # Optional perceptual loss
                perc_loss = args.perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())

                loss_g = rec_loss + gen_adv_loss + perc_loss

                if step == 0 and epoch == 0:
                    print("x1_pred:", tuple(x1_pred.shape))

            gradacc_g.step(loss_g, step)

            # ── Discriminator step ─────────────────────────────────
            with autocast(enabled=True):
                logits_fake_detach = discriminator(x1_pred.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(
                    logits_fake_detach, target_is_real=False, for_discriminator=True
                )
                logits_real = discriminator(img1.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(
                    logits_real, target_is_real=True, for_discriminator=True
                )
                loss_d = args.adv_weight * 0.5 * (d_loss_fake + d_loss_real)

            gradacc_d.step(loss_d, step)

            # ── Logging to W&B ────────────────────────────────────
            avgloss.put('Generator/rec_loss', rec_loss.item())
            avgloss.put('Generator/adv_loss', gen_adv_loss.item())
            avgloss.put('Generator/perc_loss', perc_loss.item())
            avgloss.put('Discriminator/loss', loss_d.item())
            avgloss.put('Generator/total',      loss_g.item())

            if step % 5 == 0:
                progress_bar.set_postfix({
                    "g_rec": f"{rec_loss.item():.4f}",
                    "g_adv": f"{gen_adv_loss.item():.4f}",
                    "g_perc": f"{perc_loss.item():.4f}",
                    "g_total": f"{loss_g.item():.4f}",
                    "d": f"{loss_d.item():.4f}",
                })

            if total_counter % 10 == 0:
                # Log average losses every 10 iters
                metrics = {
                    'train/gen/rec_loss': avgloss.pop_avg('Generator/rec_loss'),
                    'train/gen/adv_loss': avgloss.pop_avg('Generator/adv_loss'),
                    'train/gen/perc_loss': avgloss.pop_avg('Generator/perc_loss'),
                    'train/disc/loss': avgloss.pop_avg('Discriminator/loss'),
                    'train/Generator/total': avgloss.pop_avg('Generator/total'),
                    'step': total_counter,
                    'epoch': epoch
                }
                wandb.log(metrics, step=total_counter)

                # 2) Prepare volumes for 3×3 grid (take first element of batch, index 0)
                baseline_vol = img0[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
                true_vol     = img1[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
                pred_vol     = x1_pred[0, 0].detach().cpu().numpy()[None, ...]# [1,D,H,W]

                # 3) (Optional) also get the VAE’s reconstruction of baseline
                with torch.no_grad():
                    recon0 = autoencoder.reconstruct(img0[:1])              # [1,1,D,H,W]
                recon_vol = recon0[0, 0].detach().cpu().numpy()[None, ...]  # [1,D,H,W]

                # 4) Convert normalized ages back to months (if needed)
                age0_months = age0[0].item() * (84.0-12.0) + 12.0
                age1_months = age1[0].item() * (84.0-12.0) + 12.0

                # 5) Build meta dict including subject_id and image_uid
                meta = {
                    'subject_id':    subject_id[0],     # e.g. “sub-10098” 
                    'start_age':     age0_months,     # in years
                    'followup_age':  age1_months      # in years
                }

                log_recon_3x4_to_wandb(
                    meta,
                    baseline_vol,
                    recon_vol,
                    true_vol,
                    pred_vol,
                    step=total_counter,
                    tag="Train/Recon_Pred"
                )

                # For logging differences in original images and predictions
                # tmpd = log_differences_in_predictions(autoencoder, img0, img1, total_counter)
                # # keep the filename for the GIF
                # diff_frames.append(tmpd)
                # os.remove(tmpd)

            total_counter += 1

        # ── End of epoch: save model checkpoints ────────────────
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            cond_decoder.state_dict(),
            os.path.join(args.output_dir, f'cond_decoder_epoch{epoch}.pth')
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(args.output_dir, f'discriminator_epoch{epoch}.pth')
        )

        
        print(f"Epoch {epoch} ▶︎ Train L1 = {avgloss.get_avg('Generator/rec_loss'):.4f}")

        val_means = evaluate_one_epoch(
                    model, discriminator, valid_loader, DEVICE,
                    l1_loss_fn, perc_loss_fn, adv_loss_fn,
                    adv_weight=args.adv_weight, perceptual_weight=args.perceptual_weight,
                    autoencoder=autoencoder
                )
        
        # val_total = float(val_means["total"])
        # val_rec   = float(val_means["rec"])
        wandb.log({
            "val/loss/rec": val_means["rec"],
            "val/loss/adv": val_means["adv"],
            "val/loss/perc": val_means["perc"],
            "val/loss/total": val_means["total"],
            "val/loss/disc": val_means["d"],

            "val/ssim/pred_vs_t2": val_means["ssim_pred_t2"],
            "val/ssim/t1_vs_t2": val_means["ssim_t1_t2"],
            "val/ssim/pred_vs_t1recon": val_means["ssim_pred_t1recon"],

            "epoch": epoch
        })
        # Keep a clean "best so far" in the summary (useful for sweeps)
        wandb.run.summary["val/Generator/total"] = val_total
        print(f"Epoch {epoch} ▶︎ Val total = {val_total:.4f}  |  Val L1 = {val_rec:.4f}")

        _save_ckpt(
        last_ckpt_path, cond_decoder, discriminator,
        extra={"epoch": epoch, "val_total": float(val_means["total"])}
        )

        # Save BEST checkpoint if improved
        if _is_better(val_total, best_val, delta_abs=args.early_stop_delta):
            best_val = val_total
            best_epoch = int(epoch)
            epochs_no_improve = 0
            _save_ckpt(best_ckpt_path, cond_decoder, discriminator,
                    extra={"epoch": best_epoch, "val_total": best_val})
            wandb.log({"status/new_best": 1,
                    "status/best_epoch": best_epoch,
                    "status/best_val_total": best_val}, step=epoch)
        else:
            epochs_no_improve += 1
            wandb.log({"status/new_best": 0,
                    "status/epochs_no_improve": epochs_no_improve}, step=epoch)

        # Early stopping condition
        if (epoch + 1) >= args.min_epochs and epochs_no_improve >= args.early_stop_patience:
            print(f"[early-stop] No improvement for {epochs_no_improve} epochs "
                f"(best={best_val:.6f} @ epoch {best_epoch}). Stopping.")
            break

    summary_path = os.path.join(args.output_dir, "best_summary.json")
    summary = {
        "run_id": wandb.run.id if wandb.run else None,
        "fold": int(args.fold),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "best_ckpt_path": best_ckpt_path,
        "n_epochs_run": epoch + 1
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {summary_path}")

    # For evolution of differences during training...
    # read back all frames in order
    # frames = [Image.open(f) for f in diff_frames]
    # gif_path = "diff_evolution.gif"
    # frames[0].save(
    #     gif_path,
    #     format="GIF",
    #     save_all=True,
    #     append_images=frames[1:],
    #     duration=750,   # ms per frame
    #     loop=0          # 0 = infinite loop
    # )  

    # upload the GIF as a W&B video
    # wandb.log({ "Diffs Evolution": wandb.Video(gif_path, format="gif") })

    # # clean up
    # for fn in diff_frames:
    #     os.remove(fn)
    # os.remove(gif_path)
    # ── Finish W&B run ───────────────────────────────────────
    wandb.finish()
