import os
import argparse
import math
import subprocess
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import L1Loss
import copy
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from monai.data import MetaTensor
import wandb
from PIL import Image

from generative.losses import PerceptualLoss, PatchAdversarialLoss
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
from monai.transforms import Compose, RandGaussianNoiseD, RandBiasFieldD, RandAffined
import random
from collections import defaultdict, Counter
from torch.utils.data import Sampler
import time

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"

def t(): return time.perf_counter()

class SubjectBalancedBatchSampler(Sampler):
    """
    Each batch has subjects_per_batch * samples_per_subject == batch_size.
    Uses dataset.subject_ids / dataset.indices_by_subject if available.
    """
    def __init__(self, dataset, batch_size=3, subjects_per_batch=1,
                 samples_per_subject=3, with_replacement=True, drop_last=True):
        assert subjects_per_batch * samples_per_subject == batch_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.subjects_per_batch = subjects_per_batch
        self.samples_per_subject = samples_per_subject
        self.with_replacement = with_replacement
        self.drop_last = drop_last

        if hasattr(dataset, "indices_by_subject"):
            self.by_subj = dict(dataset.indices_by_subject)
        else:
            sids = getattr(dataset, "subject_ids", None)
            if sids is None and hasattr(dataset, "df"):
                sids = dataset.df['subject_id'].astype(str).tolist()
            if sids is None:
                raise ValueError("Dataset must expose subject_ids or df['subject_id']")
            by = defaultdict(list)
            for i, sid in enumerate(sids):
                by[sid].append(i)
            self.by_subj = dict(by)

        self.subj_keys = list(self.by_subj.keys())

    def __iter__(self):
        by_subj = {k: v[:] for k, v in self.by_subj.items()}
        for v in by_subj.values(): random.shuffle(v)
        subjects = self.subj_keys[:]
        random.shuffle(subjects)

        batch = []
        while subjects:
            chosen = [subjects.pop()]  # subjects_per_batch=1 in your case
            for s in chosen:
                pool = by_subj[s]
                if len(pool) >= self.samples_per_subject:
                    take = [pool.pop() for _ in range(self.samples_per_subject)]
                elif self.with_replacement and len(self.by_subj[s]) > 0:
                    take = pool[:]  # whatever’s left
                    while len(take) < self.samples_per_subject:
                        take.append(random.choice(self.by_subj[s]))
                    by_subj[s] = []
                else:
                    continue
                batch.extend(take)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            need = self.batch_size - len(batch)
            flat = [i for vs in self.by_subj.values() for i in vs]
            batch.extend(random.choices(flat, k=need))
            yield batch

    def __len__(self):
        N = len(self.dataset)
        return N // self.batch_size  # good enough for schedulers/grad-acc
    
train_aug = Compose([
    RandGaussianNoiseD(keys=['img'], prob=0.5, std=0.01),
    RandBiasFieldD(keys=['img'], prob=0.3, coeff_range=(0.0, 0.5)),
    RandAffined(keys=['img'], prob=0.3, rotate_range=(0.03,0.03,0.03),
                translate_range=(1,1,1), scale_range=(0.02,0.02,0.02), mode='bilinear')
])

# # @torch.no_grad()
# def enc_mu(enc, x):                     # use mean, not a stochastic sample
#     mu, _ = enc.encode(x)
#     return mu

# # --- grad-enabled, chunked encoder forward (no checkpoint needed) ---
# def encode_mu_grad_chunked(ae, x, chunk=1):
#     outs = []
#     B = x.size(0)
#     for i in range(0, B, chunk):
#         xi = x[i:i+chunk]
#         mu, _ = ae.encode(xi)          # keep grad wrt xi
#         outs.append(mu)
#     return torch.cat(outs, dim=0)      # [B, C_lat, D_lat, H_lat, W_lat]

def _to_plain(x):
    return x.as_tensor() if isinstance(x, MetaTensor) else x

# def encode_mu_grad_chunked(ae, x, chunk=1):
#     outs = []
#     for i in range(0, x.size(0), chunk):
#         xi = x[i:i+chunk]
#         def f(inp):
#             mu, _ = ae.encode(inp)
#             return mu
#         mu = checkpoint(f, xi)    # grad path, memory-friendly
#         outs.append(mu)
#     return torch.cat(outs, dim=0)

def encode_mu_grad_chunked(
    ae,
    x,                          # [B, C, D, H, W]
    batch_chunk: int = 1,       # split the batch into micro-batches
    segs: int = 4,              # desired #segments for encoder
    use_reentrant: bool = False,
    amp: bool = True,
):
    """
    Memory-reduced encode that keeps grad w.r.t. x.
    1) Splits batch into micro-chunks
    2) Optionally runs ae.encoder via checkpoint_sequential over 'segs' segments
    3) Applies ae.quant_conv and returns mu (drops logvar)
    """
    outs = []

    # Turn ae.encoder into a sequence for checkpoint_sequential
    enc_children = list(ae.encoder.children())  # top-level, safe to use
    n_funcs = len(enc_children)

    # Safe segment count
    if segs is None or segs < 1:
        segs_eff = 1
    else:
        segs_eff = min(segs, max(1, n_funcs))
        if segs_eff != segs:
            warnings.warn(f"[encode_mu_grad_chunked] Clamping segs from {segs} to {segs_eff} (encoder has {n_funcs} children).")

    # Build a Sequential only if we will actually segment
    seq = nn.Sequential(*enc_children) if (n_funcs > 0 and segs_eff > 1) else None

    B = x.size(0)
    for i in range(0, B, batch_chunk):
        xi = x[i:i+batch_chunk]

        with autocast(enabled=amp):
            if seq is not None:
                # segmented forward of the encoder body
                h = checkpoint_sequential(seq, segs_eff, xi, use_reentrant=use_reentrant)
                # finish encode path manually
                moments = ae.quant_conv(h)
                mu, _logvar = torch.chunk(moments, 2, dim=1)
            else:
                # fallback: no segmentation (also covers non-sequential encoders)
                mu, _logvar = ae.encode(xi)

        outs.append(mu)
        del xi, mu

    return torch.cat(outs, dim=0)

@torch.no_grad()
def enc_mu_nograd(ae, x):
    mu, _ = ae.encode(x)
    return mu

def enc_mu_ckpt(ae, x):
    # returns mu with grad wrt x, but uses checkpointing to save memory
    def _fn(inp):
        mu, _ = ae.encode(inp)
        return mu
    return checkpoint(_fn, x)

class SubjectTriadBatchSampler(Sampler):
    """
    Yields 3-sample batches, all from the same subject.
    - If a subject has k samples, we emit ceil(k/3) batches.
      Every real index is used exactly once per epoch.
    - For the last chunk of size 1 or 2, we replicate one item to reach 3,
      and mark those replicas with aug_flag=1 so the Dataset can augment input.
    Output per batch: [(idx, aug_flag), (idx, aug_flag), (idx, aug_flag)]
    """
    def __init__(self, dataset):
        self.dataset = dataset

        # Build per-subject index lists
        if hasattr(dataset, "indices_by_subject"):
            self.by_subj = {k: v[:] for k, v in dataset.indices_by_subject.items()}
        else:
            sids = getattr(dataset, "subject_ids", None)
            if sids is None and hasattr(dataset, "df"):
                sids = dataset.df["subject_id"].astype(str).tolist()
            if sids is None:
                raise ValueError("Dataset must expose subject_ids or df['subject_id']")
            by = defaultdict(list)
            for i, sid in enumerate(sids):
                by[sid].append(i)
            self.by_subj = dict(by)

        self.subj_keys = list(self.by_subj.keys())

    def __iter__(self):
        # shuffle subjects each epoch
        subjects = self.subj_keys[:]
        random.shuffle(subjects)

        for s in subjects:
            pool = self.by_subj[s][:]
            if len(pool) == 0:
                continue
            random.shuffle(pool)

            # walk through subject indices in chunks of 3
            for start in range(0, len(pool), 3):
                chunk = pool[start:start+3]
                n = len(chunk)
                if n == 3:
                    idxs = chunk
                    aug  = [0, 0, 0]
                elif n == 2:
                    base = random.choice(chunk)
                    idxs = [chunk[0], chunk[1], base]
                    aug  = [0, 0, 1]
                else:  # n == 1
                    base = chunk[0]
                    idxs = [base, base, base]
                    aug  = [0, 1, 1]

                yield list(zip(idxs, aug))

    def __len__(self):
        # number of triads per epoch = sum over subjects ceil(n_s/3)
        return sum(math.ceil(len(v) / 3) for v in self.by_subj.values())

# def two_view_consistency(autoencoder, x0, age0, age1, cond_pred_fn):
#     age0 = age0.to(x0.device).float()
#     age1 = age1.to(x0.device).float()
#     # view 1 (as is)
#     y1 = cond_pred_fn(x0, age0, age1)
#     z1 = enc_mu(autoencoder, y1).view(1, -1)
#     # view 2 (augmented baseline)
#     x0_aug = train_aug({'img': x0[0]})['img'].unsqueeze(0).to(x0.device)
#     y2 = cond_pred_fn(x0_aug, age0, age1)
#     z2 = enc_mu(autoencoder, y2).view(1, -1)
#     return ((z1 - z2)**2).mean()


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
    top_title = f"{sid}  |  ΔAge={(fa-sa):.2f} mo" if sa is not None and fa is not None else sid
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
        top_title += f"  |  ΔAge = {followup_age - start_age:.2f} mo"
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

# helpers for unit conversion
def norm_to_months(x_norm: torch.Tensor) -> torch.Tensor:
    # x_norm in [0,1]  -> months in [12,84]
    return 12.0 + 72.0 * x_norm

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # returns scalar tensor; NaN if mask empty (fine for wandb)
    if mask.any():
        return x[mask].mean()
    else:
        return x.new_tensor(float("nan"))

@torch.no_grad()
def tau_months_by_age0(age0_m: torch.Tensor,
                       young_cut=24.0,   # months
                       mid_cut=48.0,     # months
                       tau_young=9.0,    # months (allow more change if young)
                       tau_mid=6.0,      # months
                       tau_old=3.0):     # months (encourage identity more if older)
    """
    Piecewise tau(age0): bigger for young brains (fast development),
    smaller for older (slow changes).
    """
    a0 = age0_m
    tau = torch.where(a0 < young_cut, torch.tensor(tau_young,  device=a0.device, dtype=a0.dtype),
          torch.where(a0 < mid_cut,   torch.tensor(tau_mid,    device=a0.device, dtype=a0.dtype),
                                      torch.tensor(tau_old,    device=a0.device, dtype=a0.dtype)))
    return tau  # months

def tau_months_smooth(age0_m: torch.Tensor,
                      tau_young=9.0, tau_old=3.0,
                      pivot=36.0, steep=0.12):
    """
    Smoothly interpolate tau from tau_young at 12 mo toward tau_old at 84 mo.
    'pivot' ~ age where transition centers; 'steep' controls sharpness.
    """
    # logistic from [12..84] months mapped to ~[0..1]
    s = torch.sigmoid(steep * (age0_m - pivot))
    # when age is small (~12), s≈sigmoid(-) ~ 0 → tau≈tau_young
    # when age is large, s→1 → tau≈tau_old
    return tau_old + (tau_young - tau_old) * (1.0 - s)

def small_step_weight_months(age0, age1, *,  # ages normalized [0,1]
                             tau_mode="smooth",  # "smooth" or "piecewise"
                             # piecewise params:
                             young_cut=24.0, mid_cut=48.0,
                             tau_young=9.0, tau_mid=6.0, tau_old=3.0,
                             # smooth params:
                             pivot=36.0, steep=0.12,
                             tau_young_s=9.0, tau_old_s=3.0):
    """
    age0_norm, age1_norm: [B,1] normalized to [0,1] where 0≡12mo, 1≡84mo.
    Returns weights w in [0,1] shaped [B,1], computed in **months**.
    """
    age0_m = norm_to_months(age0)               # [B,1] months
    age1_m = norm_to_months(age1)               # [B,1] months
    d_m    = (age1_m - age0_m).abs()                 # [B,1] Δmonths

    if tau_mode == "piecewise":
        tau = tau_months_by_age0(
            age0_m.squeeze(1),
            young_cut=young_cut, mid_cut=mid_cut,
            tau_young=tau_young, tau_mid=tau_mid, tau_old=tau_old
        ).unsqueeze(1)  # [B,1]
    elif tau_mode == "smooth":
        tau = tau_months_smooth(
            age0_m.squeeze(1),
            tau_young=tau_young_s, tau_old=tau_old_s,
            pivot=pivot, steep=steep
        ).unsqueeze(1)  # [B,1]
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")
    
    w_id = torch.exp(- d_m / tau.clamp_min(1e-6))  # [B,1]
    return w_id

# 1) MLP takes 3-dim input now, triplets of age0, age1 and delta_age
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
        aug_flag = 0
        if isinstance(idx, (tuple, list)):
            # we accept (index, aug_flag)
            idx, aug_flag = idx
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

        # 3) If this item is a replicated “view”, apply augmentation to the input image (x0)
        if aug_flag == 1:
            img0 = train_aug({'img': img0})['img']

        # 4) Grab subject_id and followup_image_uid
        subject_id     = row['subject_id']            # e.g. “sub-10098”
        # image_uid      = row['followup_image_uid']    # e.g. “ses-002”

        return img0, age0, img1, age1, subject_id

def sigma_by_pair_age_months(a_pair_m: torch.Tensor,
                             cut1=24.0, cut2=48.0,
                             sig_young=6.0, sig_mid=9.0, sig_old=12.0):
    """
    Piecewise sigma(age): smaller for young (↓weight on close ages),
    larger for older (↑consistency).
    a_pair_m: [B,B] months (e.g., min or mean age of the pair)
    returns:  [B,B] months
    """
    dev, dt = a_pair_m.device, a_pair_m.dtype
    s_y = torch.tensor(sig_young, device=dev, dtype=dt)
    s_m = torch.tensor(sig_mid,   device=dev, dtype=dt)
    s_o = torch.tensor(sig_old,   device=dev, dtype=dt)
    return torch.where(
        a_pair_m < cut1, s_y,
        torch.where(a_pair_m < cut2, s_m, s_o)
    )

def within_subject_latent_loss_ageaware(
    z_pred: torch.Tensor,        # [B, D]
    age_norm: torch.Tensor,      # [B] or [B,1] in [0,1]
    subject_id,                  # list[str] or LongTensor [B]
    # age→sigma schedule (months):
    cut1=24.0, cut2=48.0,
    sig_young=6.0, sig_mid=9.0, sig_old=12.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    L = sum_{i<j, same subj} w_ij * MSE(z_i,z_j) / sum w_ij,
    with w_ij = exp(- |Δage_months| / sigma(age_pair_months))

    For 'age_pair_months' we use min(age_i, age_j): if either point is very young,
    allow more change (smaller sigma → smaller weight).
    """
    dev = z_pred.device
    B   = z_pred.size(0)

    # ages → [B] months
    if age_norm.dim() == 2 and age_norm.size(1) == 1:
        age_norm = age_norm.view(-1)
    age_m = norm_to_months(age_norm.to(dev))  # [B]

    # subject ids → LongTensor [B]
    if isinstance(subject_id, (list, tuple)):
        sids = list(map(str, subject_id))
        uniq = {s: i for i, s in enumerate(dict.fromkeys(sids))}
        subj = torch.tensor([uniq[s] for s in sids], dtype=torch.long, device=dev)
    elif isinstance(subject_id, torch.Tensor):
        subj = subject_id.to(dev).long()
    else:
        raise TypeError("subject_id must be list/tuple[str] or Tensor")

    # masks for same-subject pairs, upper triangle (no double count)
    same = subj.unsqueeze(0).eq(subj.unsqueeze(1))           # [B,B]
    tri  = torch.triu(torch.ones(B, B, dtype=torch.bool, device=dev), diagonal=1)
    mask = same & tri
    if not mask.any():
        return z_pred.new_zeros([])

    # pairwise MSE in latent
    diff = z_pred.unsqueeze(1) - z_pred.unsqueeze(0)         # [B,B,D]
    mse  = (diff * diff).mean(dim=-1)                        # [B,B]

    # age matrices (months)
    ai = age_m.unsqueeze(0)                                   # [1,B]
    aj = age_m.unsqueeze(1)                                   # [B,1]
    d_months = (ai - aj).abs()                                # [B,B]
    a_pair   = torch.minimum(ai, aj)                          # [B,B]  (use min-age in the pair)

    # sigma(age_pair) in months
    sigma_m = sigma_by_pair_age_months(a_pair, cut1, cut2, sig_young, sig_mid, sig_old)

    # weights
    w = torch.exp(- d_months / sigma_m.clamp_min(1e-6))       # [B,B]

    # weighted average over masked pairs
    num = (mse * w * mask).sum()
    den = (w   *     mask).sum().clamp_min(eps)
    return num / den


def within_subject_latent_loss(
    z_pred: torch.Tensor,          # [B, D] latent/features with grad
    age: torch.Tensor,             # [B] or [B,1] ages (same units as sigma)
    subject_id,                    # list[str] OR LongTensor of shape [B]
    sigma: float = 0.1,            # age scale for weighting 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Pairwise subject-aware latent consistency:
        L = sum_{i<j, same subj} w_ij * MSE(z_i, z_j) / sum_{i<j, same subj} w_ij
    where w_ij = exp(-|age_i - age_j| / sigma).

    Returns a scalar tensor. If no same-subject pairs exist in the batch, returns 0.
    """
    device = z_pred.device
    B = z_pred.size(0)

    # age → [B] on same device
    if age.dim() == 2 and age.size(1) == 1:
        age = age.view(-1)
    age = age.to(device)

    # subject_id → LongTensor [B] on same device
    if isinstance(subject_id, (list, tuple)):
        sids = list(map(str, subject_id))
        # stable, order-preserving mapping to ints
        uniq = {s: i for i, s in enumerate(dict.fromkeys(sids))}
        subj = torch.tensor([uniq[s] for s in sids], dtype=torch.long, device=device)
    elif isinstance(subject_id, torch.Tensor):
        subj = subject_id.to(device)
        if subj.dtype != torch.long:
            subj = subj.long()
    else:
        raise TypeError("subject_id must be a list/tuple of strings or a tensor")

    # same-subject mask, upper triangle only (no self-pairs, no double count)
    same = subj.unsqueeze(0).eq(subj.unsqueeze(1))           # [B,B]
    tri  = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
    mask = same & tri                                        # [B,B]

    if not mask.any():
        return z_pred.new_zeros([])  # scalar 0.0 (no same-subject pairs)

    # pairwise mean squared distances across feature dims
    # diff: [B,B,D] → mse: [B,B]
    diff = z_pred.unsqueeze(1) - z_pred.unsqueeze(0)
    mse  = (diff * diff).mean(dim=-1)

    # age weighting
    da   = (age.unsqueeze(0) - age.unsqueeze(1)).abs()
    w    = torch.exp(-da / max(sigma, eps))

    # weighted average over masked pairs
    num  = (mse * w * mask).sum()
    den  = (w   *     mask).sum().clamp_min(eps)
    return num / den

def global_channel_feat(mu: torch.Tensor) -> torch.Tensor:
    """
    Turns mu into [B, D] by averaging over all spatial dims (if any).
    Works for mu shaped [B,C,D,H,W] (3D), [B,C,H,W] (2D), [B,C,L] (1D), or [B,D] (flat already).
    """
    assert mu.dim() >= 2, f"mu must be at least [B, C], got {mu.shape}"
    if mu.dim() == 2:
        return mu  # already [B, D]
    reduce_dims = tuple(range(2, mu.dim()))  # all dims after channel
    return mu.mean(dim=reduce_dims)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str,
                        help="CSV with columns: split, starting_image_path, followup_image_path, starting_age, followup_age")
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--project',        default="age_conditional_decoder_sub_aware", type=str,
                        help="W&B project name")
    parser.add_argument('--run_name',       default=None,   type=str,
                        help="W&B run name (if omitted, it puts the date and time the experiment was run)")
    parser.add_argument('--fold_test',      required=True,  type=int,
                        help="Which fold (0–4) to hold out as test")
    args = parser.parse_args()

    set_determinism(0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'The device is:', DEVICE)
    print("CUDA visible:", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.version.cuda =", torch.version.cuda)
    print("torch.__version__   =", torch.__version__)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        print("nvidia-smi -L:\n", out)
    except Exception as e:
        print("nvidia-smi not runnable:", e)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # To cumulate difference plots and then create a gif
    diff_frames = []

    # Initialize WandB
    wandb.init(
        project=args.project,
        name=run_name,
        mode="offline",
        config={
            "dataset_csv": args.dataset_csv,
            "aekl_ckpt": args.aekl_ckpt,
            "disc_ckpt": args.disc_ckpt,
            "n_epochs": args.n_epochs,
            "max_batch_size": args.max_batch_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "resolution": const.RESOLUTION,
            "input_shape": const.INPUT_SHAPE_AE,
            "fold_test": args.fold_test
        }
    )
    config = wandb.config

    # ─── Build train/val splits ────────────────────────────────────
    dataset_df = pd.read_csv(config.dataset_csv)
    fold = args.fold_test
    if fold not in range(1,6):
        raise ValueError(f"fold_test must be 1..5, got {fold}")

    print(f"\n\n=== Training on folds {set(range(1,6)) - {fold}}; testing on fold {fold} ===")

    # assume `split` column holds integer fold labels 1-5
    test_df  = dataset_df[dataset_df.split == fold].reset_index(drop=True)
    train_df = dataset_df[dataset_df.split != fold].reset_index(drop=True)

    # ─── Datasets & DataLoaders ────────────────────────────────────
    train_ds = LongitudinalMRIDataset(
    df=train_df,
    resolution=config.resolution,
    target_shape=config.input_shape
    )

    batch_sampler = SubjectTriadBatchSampler(train_ds)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )


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
    age_embed_dim = 16
    cond_decoder = AgeConditionalDecoder(pretrained_ae=autoencoder, age_embed_dim=age_embed_dim).to(DEVICE)

    # Optionally unfreeze last few conv blocks if you want:  
    # for name, param in cond_decoder.decoder_body.named_parameters():  
    #     if "blocks.XX" in name:  # <-- replace XX with block index  
    #         param.requires_grad = True  

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
    k = 2
    layers = list(cond_decoder.decoder_body.children())
    for layer in layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True

    model = FullPredictor(autoencoder, cond_decoder).to(DEVICE)

    # ─── Losses & Optimizers ─────────────────────────────────────
    l1_loss_fn = L1Loss().to(DEVICE)
    # Pixel-wise reconstruction loss (|x_pred − x_true|). 
    # Drives overall anatomical fidelity and intensity accuracy.

    adv_weight = 0.025
    # Adversarial (generator) weight. Encourages realism/sharpness by fooling the discriminator.
    # Too high → flicker/instability; too low → blurry outputs.

    perceptual_weight = 0.001
    # Perceptual / feature-space similarity (e.g., VGG-like or MONAI squeeze).
    # Preserves higher-level structures and textures beyond raw pixels.
    # Reduce if training is slow or unstable.

    within_weight = 0.2
    # within_weight_start = 0.05
    # within_weight_end   = 0.5  
    # ramp_epochs         = 10
    # Within-subject latent consistency. For pairs from the same subject,
    # pulls predicted latents closer, with stronger pull when Δage is small.
    # Improves subject-specific temporal coherence.

    # two_view_weight = 0.1
    # Two-view consistency for singletons (when only one pair for a subject is in-batch).
    # Predict from x0 and an augmented x0; make their predicted latents agree.
    # Acts like self-consistency when no second timepoint from the same subject is present.

    lambda_id = 0.02   # try 0.01–0.05
    # Age-aware small-step identity regularizer.
    # When Δage is small (especially at older baseline ages), nudges prediction toward the baseline
    # (typically in latent space). Weakens automatically for younger ages where small Δage can
    # still mean noticeable anatomical change.

    # Total G loss ≈ L1 (reconstruction) + adv_weight·GAN + perceptual_weight·perceptual + within_weight·within-subject + two_view_weight·two-view + lambda_id·age-aware-identity

    kl_loss_fn  = KLDivergenceLoss()
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
        {'params': params_old, 'lr': config.lr * 0.1}
    ], weight_decay=1e-5)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr)

    # Gradient accumulation (unchanged)
    gradacc_g = GradientAccumulation(
        actual_batch_size=config.max_batch_size,
        expect_batch_size=config.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_g,
        grad_scaler=GradScaler()
    )
    gradacc_d = GradientAccumulation(
        actual_batch_size=config.max_batch_size,
        expect_batch_size=config.batch_size,
        loader_len=len(train_loader),
        optimizer=optimizer_d,
        grad_scaler=GradScaler()
    )

    avgloss = utils.AverageLoss()
    total_counter = 0

    # ─── Training Loop ───────────────────────────────────────────
    for epoch in range(config.n_epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            
            # Unpack the 5 items from LongitudinalMRIDataset.__getitem__
            img0, age0, img1, age1, subject_id = batch
            img0 = img0.to(DEVICE)   # [B,1,D,H,W]
            age0 = age0.to(DEVICE).float()
            img1 = img1.to(DEVICE)
            age1 = age1.to(DEVICE).float()   # use follow-up age
            

            # ── Generator (“conditional decoder”) step ─────────────
            with autocast(enabled=True):
                x1_pred = model(img0, age0, age1)  

                x1_small = F.avg_pool3d(_to_plain(x1_pred), 4, 4)  # stronger pooling to shrink graph
                z_pred   = encode_mu_grad_chunked(
                    autoencoder, x1_small, batch_chunk=2, segs=6, use_reentrant=False, amp=True
                ).view(x1_pred.size(0), -1)

                # Discriminator on fake → generator adv‐loss
                logits_fake = discriminator(x1_pred.contiguous().float())[-1]
                gen_adv_loss = adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False
                )

                # Reconstruction (L1) between x1_pred and img1
                rec_loss = l1_loss_fn(x1_pred.float(), img1.float())

                # Optional perceptual loss
                perc_loss = perceptual_weight * perc_loss_fn(x1_pred.float(), img1.float())

                # We skip KLD since encoder is frozen
                # kld_loss = torch.tensor(0.0).to(DEVICE)

                
                with torch.no_grad():
                    # z0     = enc_mu(autoencoder, img0).view(img0.size(0), -1)
                    x0_small = F.avg_pool3d(_to_plain(img0), 4, 4)
                    z0       = enc_mu_nograd(autoencoder, x0_small)

                # flatten BOTH to [B, Dflat]
                z_pred_feat = z_pred.view(z_pred.size(0), -1)
                z0_feat     = z0.view(z0.size(0), -1)

                if step == 0 and epoch == 0:
                    print("x1_small:", tuple(x1_small.shape))
                    # print("x1_pred:", tuple(x1_pred.shape))
                    print("z_pred_mu:", tuple(z_pred.shape))
                    print("z0_mu:", tuple(z0.shape))
                    print("z_pred_feat:", tuple(z_pred_feat.shape))

                # batch-level within-subject loss from same subject
                l_within = within_subject_latent_loss(z_pred_feat, age1.squeeze(1), subject_id, sigma=0.1)
                loss_g = rec_loss + gen_adv_loss + perc_loss + within_weight * l_within
                # l_within = within_subject_latent_loss_ageaware(z_pred_feat, age1.squeeze(1), subject_id)
                # within_weight = within_weight_start + (within_weight_end - within_weight_start) * min(epoch / ramp_epochs, 1.0)
                # loss_g = rec_loss + gen_adv_loss + perc_loss + within_weight * l_within

                # # ages in months
                # age0_m = norm_to_months(age0)           # [B,1]
                # age1_m = norm_to_months(age1)           # [B,1]
                # d_m     = (age1_m - age0_m).abs()       # [B,1]  Δmonths

                # w_id = small_step_weight_months(
                #     age0, age1, tau_mode="smooth",
                #     pivot=36.0, steep=0.12, tau_young_s=9.0, tau_old_s=3.0
                # )

                # # apply weight per-sample
                # l_id_lat_per = (z_pred_feat - z0_feat).pow(2).mean(dim=1, keepdim=True)   # [B,1]
                # l_id_lat     = (w_id * l_id_lat_per).mean()

                # # ----- per-batch scalars to log -----
                # w_id_mean     = w_id.mean()                       # scalar
                # d_m_mean      = d_m.mean()                        # scalar
                # id_contrib    = lambda_id * l_id_lat              # scalar (regularizer's contribution)

                # # age-bin masks (12–24, 24–48, 48–84 months)
                # b12_24 = (age0_m >= 12.0) & (age0_m < 24.0)
                # b24_48 = (age0_m >= 24.0) & (age0_m < 48.0)
                # b48_84 = (age0_m >= 48.0) & (age0_m <= 84.0)

                # # mean w_id per bin (may be NaN if no samples fall in a bin for this batch)
                # w12_24 = masked_mean(w_id.view(-1), b12_24.view(-1))
                # w24_48 = masked_mean(w_id.view(-1), b24_48.view(-1))
                # w48_84 = masked_mean(w_id.view(-1), b48_84.view(-1))

                # loss_g = loss_g + lambda_id * l_id_lat

            # ---- free big intermediates BEFORE backward ----
            del x1_small, logits_fake
            torch.cuda.empty_cache()
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
                loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

            gradacc_d.step(loss_d, step)

            # ── Logging to W&B ────────────────────────────────────
            avgloss.put('Generator/rec_loss', rec_loss.item())
            avgloss.put('Generator/adv_loss', gen_adv_loss.item())
            avgloss.put('Generator/perc_loss', perc_loss.item())
            avgloss.put('Generator/within',     (within_weight * l_within).item())
            avgloss.put('Generator/within_raw', l_within.item())
            # avgloss.put('Generator/two_view',   (two_view_weight * l_two_view).item())
            avgloss.put('Generator/total',      loss_g.item())
            avgloss.put('Discriminator/loss', loss_d.item())
            # avgloss.put('RegID/w_id_mean',         w_id_mean.item())
            # avgloss.put('RegID/w_id_12_24',        w12_24.item())
            # avgloss.put('RegID/w_id_24_48',        w24_48.item())
            # avgloss.put('RegID/w_id_48_84',        w48_84.item())
            # avgloss.put('RegID/dmonths_mean',      d_m_mean.item())
            # avgloss.put('RegID/id_contrib',        id_contrib.item())

            if (step % 50) == 0: torch.cuda.empty_cache()

            if step % 5 == 0:
                progress_bar.set_postfix({
                    "g_rec": f"{rec_loss.item():.4f}",
                    "g_adv": f"{gen_adv_loss.item():.4f}",
                    "g_perc": f"{perc_loss.item():.4f}",
                    "g_within": f"{(within_weight * l_within).item():.4f}",
                    # "g_2view": f"{(two_view_weight * l_two_view).item():.4f}",
                    "g_total": f"{loss_g.item():.4f}",
                    "d": f"{loss_d.item():.4f}",
                })
            # if total_counter % 50 == 0:
            #     it = total_counter 

            #     # flatten, sanitize, and move to numpy
            #     w_np = w_id.detach().float().view(-1)
            #     d_np = d_m.detach().float().view(-1)

            #     # remove non-finite just in case
            #     w_np = w_np[torch.isfinite(w_np)]
            #     d_np = d_np[torch.isfinite(d_np)]

            #     w_np = w_np.cpu().numpy()
            #     d_np = d_np.cpu().numpy()

            #     wandb.log({
            #         "hist/w_id": wandb.Histogram(w_np, num_bins=30),
            #         "hist/d_months": wandb.Histogram(d_np, num_bins=30),
            #         # handy scalars
            #         "stat/w_id_mean": float(w_np.mean()) if w_np.size else 0.0,
            #         "stat/w_id_min":  float(w_np.min())  if w_np.size else 0.0,
            #         "stat/w_id_max":  float(w_np.max())  if w_np.size else 0.0,
            #         "stat/dm_mean":   float(d_np.mean()) if d_np.size else 0.0,
            #     }, step=it)

            if total_counter % 10 == 0:
                it = total_counter #// 10
                # Log average losses every 10 iters
                metrics = {
                    'train/gen/rec_loss': avgloss.pop_avg('Generator/rec_loss'),
                    'train/gen/adv_loss': avgloss.pop_avg('Generator/adv_loss'),
                    'train/gen/perc_loss': avgloss.pop_avg('Generator/perc_loss'),
                    'train/gen/within':    avgloss.pop_avg('Generator/within'),
                    # 'train/gen/two_view':  avgloss.pop_avg('Generator/two_view'),
                    'train/gen/total':     avgloss.pop_avg('Generator/total'),
                    'train/disc/loss': avgloss.pop_avg('Discriminator/loss'),
                    # 'train/regid/w_id_mean':    avgloss.pop_avg('RegID/w_id_mean'),
                    # 'train/regid/w_id_12_24':   avgloss.pop_avg('RegID/w_id_12_24'),
                    # 'train/regid/w_id_24_48':   avgloss.pop_avg('RegID/w_id_24_48'),
                    # 'train/regid/w_id_48_84':   avgloss.pop_avg('RegID/w_id_48_84'),
                    # 'train/regid/dmonths_mean': avgloss.pop_avg('RegID/dmonths_mean'),
                    # 'train/regid/id_contrib':   avgloss.pop_avg('RegID/id_contrib'),
                    'step': it,
                    'epoch': epoch
                }
                wandb.log(metrics, step=it)

                # 2) Prepare volumes for 3×3 grid (take first element of batch, index 0)
                baseline_vol = img0[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
                true_vol     = img1[0, 0].detach().cpu().numpy()[None, ...]   # [1,D,H,W]
                pred_vol     = x1_pred[0, 0].detach().cpu().numpy()[None, ...]# [1,D,H,W]

                # 3) (Optional) also get the VAE’s reconstruction of baseline
                with torch.no_grad():
                    recon0 = autoencoder.reconstruct(img0[:1])              # [1,1,D,H,W]
                recon_vol = recon0[0, 0].detach().cpu().numpy()[None, ...]  # [1,D,H,W]

                # 4) Convert normalized ages back to months 
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
                    step=it,
                    tag="Train/Recon_Pred"
                )

                # 1) Compute recon_t0, recon_t1
                # with torch.no_grad():
                #     recon0 = autoencoder.reconstruct(img0[:1])
                #     recon1 = autoencoder.reconstruct(img1[:1])
                # recon_t0 = recon0[0,0].cpu().numpy()[None]
                # recon_t1 = recon1[0,0].cpu().numpy()[None]

                # 2) Compute diffs and log them
                # D1 = np.abs(pred_vol[0] - true_vol[0]) #T2pred-T2true
                # D2 = np.abs(pred_vol[0] - baseline_vol[0]) #T2pred-T1
                # D3 = np.abs(pred_vol[0] - recon_t0[0]) #T2pred-T1recon
                # D4 = np.abs(true_vol[0] - baseline_vol[0]) #T2true-T1
                # D5 = np.abs(recon_t1[0] - recon_t0[0]) #T2recon-T1recon
                # m = max(D.max() for D in (D1,D2,D3,D4,D5))
                # iz = D1.shape[0]//2
                # # make a 1×5 figure
                # fig, axes = plt.subplots(1, 5, figsize=(15,4), constrained_layout=True)
                # titles = ["|T2pred–T2true|","|T2pred–T1|","|T2pred–T1recon|","|T2true–T1|","|T2recon–T1recon|"]
                # for ax, D, ttl in zip(axes, (D1,D2,D3,D4,D5), titles):
                #     im = ax.imshow(np.rot90(D[iz]), cmap="hot", vmin=0, vmax=m)
                #     ax.set_title(ttl, fontsize=9)
                #     ax.axis("off")

                # # add single colorbar on the right
                # cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
                # cbar.set_label("Abs. Error", rotation=90)

                # delta_age = age1_months - age0_months
                # fig.suptitle(f"Epoch {epoch} | {subject_id[0]}  |  ΔAge={delta_age:.2f} mo", fontsize=12)
                # # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # # save + log
                # tmpd = f"{args.output_dir}/abs_errors_e{epoch}_it{it}.png"
                # fig.savefig(tmpd, dpi=100)
                # plt.close(fig)

                # # log to W&B
                # wandb.log({ "Errors/5-way": wandb.Image(tmpd) }, step=it)

                # # keep the filename for the GIF
                # diff_frames.append(tmpd)
                

                # 3) If you want mean‐absolute‐errors of any pair:
                # mae_pred_true   = float(D1.mean())   # for T2pred–T2true
                # mae_pred_t1     = float(D2.mean())
                # mae_pred_t1rec  = float(D3.mean())
                # mae_true_t1     = float(D4.mean())
                # mae_recon_t1rec = float(D5.mean())

                # wandb.log({
                #     "mae/T2pred-T2true":      mae_pred_true,
                #     "mae/T2pred-T1":        mae_pred_t1,
                #     "mae/T2pred-T1recon":   mae_pred_t1rec,
                #     "mae/T2true-T1":        mae_true_t1,
                #     "mae/T2recon-T1recon":         mae_recon_t1rec,
                #     "delta_age":          delta_age
                # }, step=it)

                # 6) Log 3×3 grid of [baseline, true follow-up, pred follow-up]
                # log_recon_comparison_to_wandb(
                #     meta,
                #     baseline_vol,   # [1,D,H,W]
                #     true_vol,       # [1,D,H,W]
                #     pred_vol,       # [1,D,H,W]
                #     step=it,
                #     tag="Train/Reconstruction_Comparison"
                # )

                # # 7) (Optional) Log “VAE recon vs pred” in a separate grid:
                # log_recon_comparison_to_wandb(
                #     meta,
                #     recon_vol,      # [1,D,H,W] ← VAE’s baseline reconstruction
                #     true_vol,
                #     pred_vol,
                #     step=it,
                #     tag="Train/VAE_vs_Pred"
                # )

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

    # # upload the GIF as a W&B video
    # wandb.log({ "Diffs Evolution": wandb.Video(gif_path, format="gif") })

    # # clean up
    # for fn in diff_frames:
    #     os.remove(fn)
    # os.remove(gif_path)

    # ── Finish W&B run ───────────────────────────────────────
    wandb.finish()
