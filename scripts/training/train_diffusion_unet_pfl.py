import os
import argparse
import csv

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm

from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd,
    sample_using_diffusion
)


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    _dict['context'] = torch.tensor([ _dict[c] for c in const.CONDITIONING_VARIABLES ]).unsqueeze(0)
    return _dict


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """

    for tag_i, size in enumerate([ 'small', 'medium', 'large' ]):

        # context = torch.tensor([[
        #     (torch.randint(60, 99, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age 
        #     (torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
        #     (torch.randint(1, 3,   (1,)) - const.DIA_MIN) / const.DIA_DELTA,  # diagnosis
        #     0.567, # (mean) cerebral cortex 
        #     0.539, # (mean) hippocampus
        #     0.578, # (mean) amygdala
        #     0.558, # (mean) cerebral white matter
        #     0.30 * (tag_i+1), # variable size lateral ventricles
        # ]])

        context = torch.tensor([[
            (torch.randint(1, 7, (1,)) - const.AGE_MIN) / const.AGE_DELTA,  # age 
            (torch.randint(1, 2,   (1,)) - const.SEX_MIN) / const.SEX_DELTA,  # sex
            0.573, # (mean) cerebral cortex 
            0.513, # (mean) hippocampus
            0.508, # (mean) amygdala
            0.474, # (mean) cerebral white matter
            0.155 * (tag_i+1), # variable size lateral ventricles
        ]])

        image = sample_using_diffusion(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            context=context,
            device=DEVICE, 
            scale_factor=scale_factor
        )

        utils.tb_display_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/{size}_ventricles',
            image=image
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',  required=True, type=str)
    parser.add_argument('--cache_dir',  required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--aekl_ckpt',  required=True, type=str)
    parser.add_argument('--diff_ckpt',   default=None, type=str)
    parser.add_argument('--num_workers', default=8,     type=int)
    parser.add_argument('--n_epochs',    default=5,     type=int)
    parser.add_argument('--batch_size',  default=16,    type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    args = parser.parse_args()
    
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys=['latent_path'], names=['latent']),
        transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
        transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates)
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[ dataset_df.split == 'train' ]
    valid_df = dataset_df[ dataset_df.split == 'valid' ]
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True,
                              pin_memory=True)
    
    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              persistent_workers=True, 
                              pin_memory=True)
    
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['latent']
    print(f"Latent shape: {z.shape}")

    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")


    writer = SummaryWriter()
    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }

    all_losses = []


    for epoch in range(args.n_epochs):

        epoch_losses = {}
        epoch_steps = {}

        for mode in loaders.keys():
            
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast(enabled=True):    
                        
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    # --- pull out baseline and follow‐up latents ---
                    z1 = batch['latent_t1'].to(DEVICE) * scale_factor      # [B, C, D, H, W]
                    z2 = batch['latent_t2'].to(DEVICE) * scale_factor      # [B, C, D, H, W]
                    n  = z2.shape[0]

                    # --- build context from target age (and any other const.CONDITIONING_VARIABLES) ---
                    # e.g. if CONDITIONING_VARIABLES = ['age_t2','sex','dia',…] then:
                    raw = [ batch[k] for k in const.CONDITIONING_VARIABLES ]
                    # each raw[i] is shape [B], so stack → [num_vars, B] → transpose → [B, num_vars]
                    context = torch.cat([context, z1.mean(dim=[2,3,4])], dim=1)           
                    # (if age needs normalization, do it here: (age_t2 - AGE_MIN)/AGE_DELTA )

                    with torch.set_grad_enabled(mode == 'train'):

                        # --- forward‐diffuse the true follow‐up latent z2 ---
                        noise     = torch.randn_like(z2)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                        x_t       = scheduler.add_noise(z2, noise, timesteps)

                        # --- predict that noise (conditioned on age2 ± z1 if you choose) ---
                        # If you also want to give the network z1, you can concatenate it
                        # into the context or feed it as a separate cross‐attention key.
                        noise_pred = inferer(
                            inputs=x_t,                       # already scaled
                            diffusion_model=diffusion,
                            timesteps=timesteps,
                            condition=context,                # [B, num_vars]
                            mode='crossattn'
                        )

                        # --- classic LDM loss ---
                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(loader)

            epoch_losses[mode] = epoch_loss
            epoch_steps[mode] = len(loader)

            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
            images_to_tensorboard(
                writer=writer, 
                epoch=epoch, 
                mode=mode, 
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                scale_factor=scale_factor
            )

        all_losses.append({
        'epoch': epoch,
        'train_loss': epoch_losses.get('train'),
        'train_steps': epoch_steps.get('train'),
        'valid_loss': epoch_losses.get('valid'),
        'valid_steps': epoch_steps.get('valid'),
        })

        # save the model                
        savepath = os.path.join(args.output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diffusion.state_dict(), savepath)

        #save csv
        csv_path = os.path.join(args.output_dir, "epoch_losses.csv")
        write_header = epoch == 0
        with open(csv_path, mode="a", newline="") as f:
            fieldnames = ["epoch", "train_loss", "train_steps", "valid_loss", "valid_steps"]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                csv_writer.writeheader()
            csv_writer.writerow(all_losses[-1])