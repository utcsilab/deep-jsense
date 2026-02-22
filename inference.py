#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob

import matplotlib.pyplot as plt
import torch

from datagen import MCFullFastMRI, crop
from models import MoDLDoubleUnroll
from utils import ifft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Validation data
core_dir  = '/path/to/multicoil_val'
val_files = sorted(glob.glob(core_dir + '/*.h5'))

# Load pretrained model
filename = "pretrained_models/knee.pt"
contents = torch.load(filename, map_location=device)
hparams = contents['hparams']

model = MoDLDoubleUnroll(hparams)
model = model.to(device)
model.load_state_dict(contents["model_state_dict"], strict=False)
model.eval()

# Validation dataset
center_slice = 15 
num_slices = 5 
val_dataset = MCFullFastMRI(val_files, num_slices, center_slice,
                              use_acs=hparams.use_acs, acs_lines=hparams.acs_lines,
                              downsample=hparams.downsample, scramble=True, 
                              mps_kernel_shape=hparams.mps_kernel_shape,
                              maps=None)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)
iterator = iter(val_loader)

# Run through some samples (in random slice order)
for sample_idx in range(4):
    sample = next(iterator)
    if sample['ksp'].shape[-2] < 320:
        print('Skipping small scan!')
        continue
    if sample['ksp'].shape[-2] > 348:
        print('Skipping large scan!')
        continue

    # Move to device
    for key, value in sample.items():
        try:
            sample[key] = sample[key].to(device)
        except:
            pass
            
    # Get outputs
    with torch.inference_mode():
        # Estimate
        est_img_kernel, est_map_kernel, est_ksp = \
            model(sample, hparams.meta_unrolls_end)
        
        # Extra padding with dead zones
        est_ksp_padded = torch.nn.functional.pad(est_ksp, (
            torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
            torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))
        
        # Convert to image domain
        est_img_coils = ifft(est_ksp_padded)
        
        # RSS images
        est_img_rss = torch.sqrt(torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))
        est_crop_rss = crop(est_img_rss, 320, 320)
        # Plot
        plt.imshow(est_crop_rss[0].cpu(), cmap='gray')
        plt.axis('off')
        plt.savefig(f"inference_sample{sample_idx}.png", dpi=300, bbox_inches="tight")
        plt.close()
