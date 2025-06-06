import os
import argparse
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter

from brlp import const
from brlp import utils
from brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd
)

# ----------------------------------------------------------------------------
# Training script with Supervised Contrastive Loss & Age-based Latent Smoothing
# ----------------------------------------------------------------------------

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      required=True, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--aug_p',          default=0.8,   type=float)
    # SupCon hyperparameters
    parser.add_argument('--lambda_sc',      default=0.1,   type=float, help='Weight for SupCon loss')
    parser.add_argument('--tau_supcon',     default=0.07,  type=float, help='Temperature for SupCon')
    # Age-based smoothing hyperparameters
    parser.add_argument('--lambda_age',     default=0.1,   type=float, help='Weight for age-based latent loss')
    parser.add_argument('--tau_age',        default=5.0,   type=float, help='Temperature for age similarity')
    args = parser.parse_args()

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[dataset_df.split == 'train']
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.max_batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True
    )

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    # Fixed weights
    adv_weight        = 0.025
    perceptual_weight = 0.001
    kl_weight         = 1e-7

    # Loss functions
    l1_loss_fn  = L1Loss()
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

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

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
    writer = SummaryWriter()
    total_counter = 0

    eps = 1e-6
    lambda_sc  = args.lambda_sc
    tau_sc     = args.tau_supcon
    lambda_age = args.lambda_age
    tau_age    = args.tau_age

    for epoch in range(args.n_epochs):
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:
            images = batch['image'].to(DEVICE)
            # Expecting age and subject_id in batch
            ages = batch.get('age', None)
            if ages is not None:
                ages = ages.to(DEVICE).float()

            subject_ids = batch.get('subject_id', None)
            if subject_ids is not None:
                subject_ids = torch.tensor(subject_ids, device=DEVICE, dtype=torch.long)

            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_adv  = adv_weight * adv_loss_fn(
                    logits_fake, target_is_real=True, for_discriminator=False
                )

                # ---------------------------------------------------------
                # Supervised Contrastive Loss (SupCon)
                if subject_ids is not None:
                    z_flat = z_mu.view(z_mu.size(0), -1)  # [N, d]
                    sim_matrix = torch.matmul(z_flat, z_flat.T) / tau_sc  # [N, N]
                    exp_sim = torch.exp(sim_matrix)                         # [N, N]

                    # positive mask: same subject_id, exclude diagonal
                    mask = (subject_ids.unsqueeze(1) == subject_ids.unsqueeze(0)).float()
                    mask.fill_diagonal_(0)

                    # numerator and denominator
                    num = (exp_sim * mask).sum(dim=1)           # [N]
                    denom = exp_sim.sum(dim=1) - torch.diagonal(exp_sim)  # [N]
                    num   = torch.clamp(num,   min=eps)
                    denom = torch.clamp(denom, min=eps)
                    loss_sc = -torch.log(num / denom).mean()
                else:
                    loss_sc = torch.tensor(0.0, device=DEVICE)

                sc_loss = lambda_sc * loss_sc
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # Age-based latent smoothing loss
                if ages is not None:
                    z_flat = z_mu.view(z_mu.size(0), -1)  # [N, d]
                    age_diff = torch.abs(ages.unsqueeze(1) - ages.unsqueeze(0))  # [N, N]
                    age_w = torch.exp(-age_diff / tau_age)                          # [N, N]
                    age_w.fill_diagonal_(0)

                    dist_sq = torch.cdist(z_flat, z_flat, p=2) ** 2  # [N, N]
                    loss_age_raw = (age_w * dist_sq).sum() / (z_flat.size(0) * (z_flat.size(0) - 1) + eps)
                    loss_age = lambda_age * loss_age_raw
                else:
                    loss_age_raw = torch.tensor(0.0, device=DEVICE)
                    loss_age = torch.tensor(0.0, device=DEVICE)
                # ---------------------------------------------------------

                # Total generator loss
                loss_g = rec_loss + kld_loss + per_loss + gen_adv + sc_loss + loss_age

            gradacc_g.step(loss_g, step)

            # ----------------------------------------------------------------------------
            # Discriminator update (standard real/fake)
            with autocast(enabled=True):
                logits_fake_detach = discriminator(reconstruction.detach())[-1]
                d_loss_fake = adv_loss_fn(
                    logits_fake_detach, target_is_real=False, for_discriminator=True
                )
                logits_real = discriminator(images.detach())[-1]
                d_loss_real = adv_loss_fn(
                    logits_real, target_is_real=True, for_discriminator=True
                )
                loss_d = adv_weight * 0.5 * (d_loss_fake + d_loss_real)

            gradacc_d.step(loss_d, step)
            # ----------------------------------------------------------------------------

            # Logging
            avgloss.put('Generator/reconstruction_loss', rec_loss.item())
            avgloss.put('Generator/kl_regularization',   kld_loss.item())
            avgloss.put('Generator/perceptual_loss',     per_loss.item())
            avgloss.put('Generator/adversarial_loss',    gen_adv.item())
            avgloss.put('Generator/supcon_loss',         loss_sc.item())
            avgloss.put('Generator/age_raw_loss',        loss_age_raw.item())
            avgloss.put('Generator/age_loss',            loss_age.item())
            avgloss.put('Discriminator/adversarial_loss', loss_d.item())

            if total_counter % 10 == 0:
                step_tb = total_counter // 10
                avgloss.to_tensorboard(writer, step_tb)
                utils.tb_display_reconstruction(
                    writer, step_tb,
                    images[0].detach().cpu(),
                    reconstruction[0].detach().cpu()
                )

            total_counter += 1

        # Save after epoch
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),    os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))

    writer.close()
