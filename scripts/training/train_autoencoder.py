import os
import argparse
import warnings
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from monai import transforms
from datetime import datetime
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


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def wb_log_reconstruction(step: int, image: torch.Tensor, recon: torch.Tensor):
    """
    Log a 2×3 grid of orthogonal slices (original vs reconstruction)
    to Weights & Biases at the given step.
    """
    # Convert 4D (1×C×D×H×W) → 3D (D×H×W) if needed
    if image.ndim == 5:  # e.g. [1,1,D,H,W]
        image = image.squeeze(0).squeeze(0)
    if recon.ndim == 5:
        recon = recon.squeeze(0).squeeze(0)

    # Compute center indices
    d, h, w = image.shape
    md, mh, mw = d // 2, h // 2, w // 2

    # Build the figure
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    for ax in axes.flatten():
        ax.axis('off')

    # Row 0: original slices
    axes[0, 0].set_title('original (axial)',    color='cyan')
    axes[0, 0].imshow(image[md, :, :], cmap='gray')
    axes[0, 1].imshow(image[:, mh, :], cmap='gray')
    axes[0, 2].imshow(image[:, :, mw], cmap='gray')

    # Row 1: reconstructed slices
    axes[1, 0].set_title('recon (axial)',       color='magenta')
    axes[1, 0].imshow(recon[md, :, :], cmap='gray')
    axes[1, 1].imshow(recon[:, mh, :], cmap='gray')
    axes[1, 2].imshow(recon[:, :, mw], cmap='gray')

    plt.tight_layout()

    # Log to W&B
    wandb.log({"Reconstruction": wandb.Image(fig)}, step=step)

    # Clean up
    plt.close(fig)


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
    parser.add_argument('--fold_test',      required=True,  type=int,
                        help="Which fold (0–4) to hold out as test")
    args = parser.parse_args()

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    wandb.init(
    project="ae-training",
    config=vars(args),      # logs all CLI args as hyperparameters
    mode="offline",         # records locally; you can later `wandb sync`
    name=f"{run_name}_foldtest{args.fold_test}",
    dir=args.output_dir     # where to write the wandb files
    )

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    # pick your test‐fold from the CLI
    fold = args.fold_test
    if fold not in range(5):
        raise ValueError(f"fold_test must be 0..4, got {fold}")

    print(f"\n\n=== Training on folds {set(range(5)) - {fold}}; testing on fold {fold} ===")

    dataset_df = pd.read_csv(args.dataset_csv)
    # train_df = dataset_df[ dataset_df.split == 'train' ]
    # split by your fold column
    train_df = dataset_df[ dataset_df['split'] != fold ].reset_index(drop=True)
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.max_batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    autoencoder   = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    adv_weight          = 0.025
    perceptual_weight   = 0.001
    kl_weight           = 1e-7

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(DEVICE)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)


    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())

    avgloss = utils.AverageLoss()
    # writer = SummaryWriter()
    total_counter = 0


    for epoch in range(args.n_epochs):
        
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            with autocast(enabled=True):

                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
                
            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Generator/reconstruction_loss',    rec_loss.item())
            avgloss.put('Generator/perceptual_loss',        per_loss.item())
            avgloss.put('Generator/adverarial_loss',        gen_loss.item())
            avgloss.put('Generator/kl_regularization',      kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss',    loss_d.item())

            
            if total_counter % 10 == 0:
                # log scalar losses
                step = total_counter // 10
                scalars = avgloss.to_dict()  # e.g. {'Generator/reconstruction_loss': ..., ...}
                wandb.log(scalars, step=step)
                wb_log_reconstruction(step=step, image=images[0].detach().cpu(), recon=reconstruction[0].detach().cpu())
                # step = total_counter // 10
                # avgloss.to_tensorboard(writer, step)
                # utils.tb_display_reconstruction(writer, step, images[0].detach().cpu(), reconstruction[0].detach().cpu())
        
            total_counter += 1

        # Save the model after each epoch.
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))
        # torch.save(discriminator.state_dict(),
        #            os.path.join(args.output_dir, f'disc_foldtest_{fold}_ep{epoch}.pth'))
        # torch.save(autoencoder.state_dict(),
        #            os.path.join(args.output_dir, f'ae_foldtest_{fold}_ep{epoch}.pth'))
