#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
# !!! SET YOUR PATHS TO BART HERE
os.environ["TOOLBOX_PATH"]    = '~/research/bart-0.8.00'
sys.path.append('~/research/bart-0.8.00/python')     

import h5py, glob
import numpy as np
from tqdm import tqdm

from bart import bart

# Load list of files
train_dir   = '/mnt/hdd15/marius/%s' % core_mode
train_files = glob.glob(core_dir + '/*.h5')

# Estimate maps for a limited 'num_slices' around 'center_slice'
center_slice = 15
num_slices   = 5
# Estimate maps using a limited number of 'acs_lines'
acs_lines = 12

# Outputs
output_dir = '%s_Wc0_Espirit_maps_acs%d' % (core_mode, acs_lines)
os.makedirs(output_dir, exist_ok=True)

# For each file and slice
for sample_idx in tqdm(range(len(train_files))):
    # Load entire volume
    with h5py.File(train_files[sample_idx], 'r') as contents:
        # Get number of slices and shape
        (_, num_coils, img_h, img_w) = contents['kspace'].shape
        
    # Output maps
    s_maps = np.zeros((num_slices, num_coils, img_h, img_w),
                      dtype=np.complex64)
    
    # For each slice
    for idx in tqdm(range(num_slices)):
        # Get central slice index
        slice_idx = center_slice + \
            np.mod(idx, num_slices) - num_slices // 2
        
        # Load specific slice
        with h5py.File(train_files[sample_idx], 'r') as contents:
            # Get k-space and s-maps for specific slice
            k_image = np.asarray(contents['kspace'][slice_idx])
        
        # ACS
        num_slices = k_image.shape[-1]
        acs_idx    = np.arange(num_slices//2 - acs_lines//2,
                               num_slices//2 + acs_lines//2)
        # Remove everything except ACS
        k_down = np.zeros(k_image.shape)
        k_down[..., acs_idx] = k_image[..., acs_idx]
        
        # Estimate maps
        s_maps[idx] = bart(1, 'ecalib -m1 -W -c0', 
                    k_down.transpose((1, 2, 0))[None,...]).transpose(
                        (3, 1, 2, 0)).squeeze()
        
    # Output filename
    output_file = output_dir + '/%s' % (os.path.basename(train_files[sample_idx]))
    
    # Write file
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('s_maps', data=s_maps.astype(np.complex64))
        hf.create_dataset('original_slice_idx', data=slice_idx.astype(np.int))