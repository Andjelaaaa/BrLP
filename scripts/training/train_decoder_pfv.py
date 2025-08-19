import os
import argparse
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
    tmp = f"tmp_{sid}_{uid}_3x4.png"
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


# ── Define AgeEmbedMLP and AgeConditionalDecoder exactly as before ──

class AgeEmbedMLP(nn.Module):
    """
    Simple two‐layer MLP: takes scalar age ∈ [0,1] → age_embed_dim vector.
    """
    def __init__(self, age_embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(age_embed_dim, age_embed_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, age_scalar):
        # age_scalar: [B,1]
        return self.mlp(age_scalar)    # → [B, age_embed_dim]

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
    
# ─── Define FullPredictor (encoder frozen + cond_decoder) ───────
class FullPredictor(nn.Module):
    def __init__(self, encoder, cond_decoder):
        super().__init__()
        self.encoder = encoder       # frozen
        self.cond_decoder = cond_decoder

    def forward(self, x0, age_target):
        # x0: [B,1,D,H,W], age_target: [B,1]
        with torch.no_grad():
            z_mu, z_sigma = self.encoder.encode(x0)
            z = self.encoder.sampling(z_mu, z_sigma)  # [B, latent_channels, D_lat, H_lat, W_lat]
        x1_pred = self.cond_decoder(z, age_target)    # [B,1,D,H,W]
        return x1_pred

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
        age0 = (row['starting_age'] * 6 - 1.0) / 6.0
        age1 = (row['followup_age'] * 6  - 1.0) / 6.0
        age0 = torch.tensor([age0], dtype=torch.float32)
        age1 = torch.tensor([age1], dtype=torch.float32)

        # 3) Grab subject_id and followup_image_uid
        subject_id     = row['subject_id']            # e.g. “sub-10098”
        # image_uid      = row['followup_image_uid']    # e.g. “ses-002”

        return img0, age0, img1, age1, subject_id

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str,
                        help="CSV with columns: split, starting_image_path, followup_image_path, starting_age, followup_age")
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--project',        default="age_conditional_decoder", type=str,
                        help="W&B project name")
    parser.add_argument('--run_name',       default=None,   type=str,
                        help="W&B run name (if omitted, it puts the date and time the experiment was run)")
    parser.add_argument('--fold_test',      required=True,  type=int,
                        help="Which fold (0–4) to hold out as test")
    args = parser.parse_args()

    set_determinism(0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            "cache_dir": args.cache_dir,
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
    if not 0 <= fold < 5:
        raise ValueError(f"--fold_test must be between 0 and 4, got {fold}")

    # assume `split` column holds integer fold labels 0–4
    test_df  = dataset_df[dataset_df.split == fold].reset_index(drop=True)
    train_df = dataset_df[dataset_df.split != fold].reset_index(drop=True)

    print(f"Training on folds {set(range(5)) - {fold}}; testing on fold {fold}")

    # ─── Datasets & DataLoaders ────────────────────────────────────
    train_ds = LongitudinalMRIDataset(
        df=train_df,
        resolution=config.resolution,
        target_shape=config.input_shape
    )
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config.max_batch_size,
        shuffle=True,
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
    adv_weight = 0.025
    perceptual_weight = 0.001
    kl_weight = 1e-7

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
            img1 = img1.to(DEVICE)
            age1 = age1.to(DEVICE)   # use follow-up age

            # ── Generator (“conditional decoder”) step ─────────────
            with autocast(enabled=True):
                x1_pred = model(img0, age1)  # [B,1,D,H,W]

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
                kld_loss = torch.tensor(0.0).to(DEVICE)

                loss_g = rec_loss + gen_adv_loss + perc_loss + kld_loss

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
            avgloss.put('Discriminator/loss', loss_d.item())

            if total_counter % 10 == 0:
                it = total_counter // 10
                # Log average losses every 10 iters
                metrics = {
                    'train/gen/rec_loss': avgloss.pop_avg('Generator/rec_loss'),
                    'train/gen/adv_loss': avgloss.pop_avg('Generator/adv_loss'),
                    'train/gen/perc_loss': avgloss.pop_avg('Generator/perc_loss'),
                    'train/disc/loss': avgloss.pop_avg('Discriminator/loss'),
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

                # 4) Convert normalized ages back to years (if needed)
                age0_years = age0[0].item() * 6.0 + 1.0
                age1_years = age1[0].item() * 6.0 + 1.0

                # 5) Build meta dict including subject_id and image_uid
                meta = {
                    'subject_id':    subject_id[0],     # e.g. “sub-10098” 
                    'start_age':     age0_years,     # in years
                    'followup_age':  age1_years      # in years
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

                delta_age = age1_years - age0_years
                fig.suptitle(f"Epoch {epoch} | {subject_id[0]}  |  ΔAge={delta_age:.2f} yr", fontsize=12)
                # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # save + log
                tmpd = f"abs_errors_e{epoch}_it{it}.png"
                fig.savefig(tmpd, dpi=100)
                plt.close(fig)

                # log to W&B
                wandb.log({ "Errors/5-way": wandb.Image(tmpd) }, step=it)

                # keep the filename for the GIF
                diff_frames.append(tmpd)
                
                # tmpd = f"tmp_err_{subject_id[0]}_{it}.png"
                # fig.savefig(tmpd, dpi=100); plt.close(fig)
                # wandb.log({"Errors/5-map": wandb.Image(tmpd)}, step=it)
                # os.remove(tmpd)

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
                }, step=it)

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
    frames = [Image.open(f) for f in diff_frames]
    gif_path = "diff_evolution.gif"
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=750,   # ms per frame
        loop=0          # 0 = infinite loop
    )  

    # upload the GIF as a W&B video
    wandb.log({ "Diffs Evolution": wandb.Video(gif_path, format="gif") })

    # clean up
    for fn in diff_frames:
        os.remove(fn)
    os.remove(gif_path)
    # ── Finish W&B run ───────────────────────────────────────
    wandb.finish()
