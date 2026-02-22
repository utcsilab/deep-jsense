#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, os
import numpy as np
import sigpy as sp
import torch

from torch.utils.data import Dataset


def crop(variable, tw, th):
    w, h = variable.shape[-2:]
    x1 = int(np.ceil((w - tw) / 2.0))
    y1 = int(np.ceil((h - th) / 2.0))
    return variable[..., x1 : x1 + tw, y1 : y1 + th]


def crop_cplx(variable, tw, th):
    w, h = variable.shape[-3:-1]
    x1 = int(np.ceil((w - tw) / 2.0))
    y1 = int(np.ceil((h - th) / 2.0))
    return variable[..., x1 : x1 + tw, y1 : y1 + th, :]


# Multicoil fastMRI dataset with various options
class MCFullFastMRI(Dataset):
    def __init__(
        self,
        sample_list,
        num_slices,
        center_slice,
        downsample,
        saved_masks=None,
        use_acs=True,
        scramble=False,
        acs_lines=4,
        mps_kernel_shape=None,
        maps=None,
        direction="y",
    ):
        self.saved_masks = saved_masks  # Pre-generated sampling masks
        self.sample_list = sample_list
        self.num_slices = num_slices
        self.center_slice = center_slice
        self.downsample = downsample
        self.mps_kernel_shape = mps_kernel_shape
        self.use_acs = use_acs
        self.acs_lines = acs_lines
        self.scramble = scramble  # Scramble samples or not?
        self.maps = maps  # Pre-estimated sensitivity maps
        self.direction = direction  # Which direction are lines in
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))

    def __len__(self):
        return len(self.sample_list) * self.num_slices

    def __getitem__(self, idx):
        # Permute if desired
        if self.scramble:
            idx = self.permute[idx]

        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2
        map_idx = np.mod(idx, self.num_slices)  # Maps always count from zero

        # Load MRI image
        with h5py.File(self.sample_list[sample_idx], "r") as contents:
            # Get k-space for specific slice
            k_image = np.asarray(contents["kspace"][slice_idx])
            ref_rss = np.asarray(contents["reconstruction_rss"][slice_idx])
            # Store core file
            core_file = os.path.basename(self.sample_list[sample_idx])
            core_slice = slice_idx

        # If desired, load external sensitivity maps
        if not self.maps is None:
            with h5py.File(self.maps[sample_idx], "r") as contents:
                # Get sensitivity maps for specific slice
                s_maps = np.asarray(contents["s_maps"][map_idx])
                s_maps_full = np.copy(s_maps)
        else:
            # Dummy values
            s_maps, s_maps_full = np.asarray([0.0]), np.asarray([0.0])

        # Compute sum-energy of lines
        # !!! This is because some lines are exact zeroes
        line_energy = np.sum(np.square(np.abs(k_image)), axis=(0, 1))
        dead_lines = np.where(line_energy < 1e-16)[0]  # Sufficient based on data

        # Always remove an even number of lines to keep original centering
        dead_lines_front = np.sum(dead_lines < 160)
        dead_lines_back = np.sum(dead_lines > 160)
        if np.mod(dead_lines_front, 2):
            dead_lines = np.delete(dead_lines, 0)
        if np.mod(dead_lines_back, 2):
            dead_lines = np.delete(dead_lines, -1)

        # Store all GT data
        gt_ksp = np.copy(k_image)

        # Remove dead lines completely
        k_image = np.delete(k_image, dead_lines, axis=-1)
        # Remove them from the frequency representation of the maps as well
        if not self.maps is None:
            k_maps = sp.fft(s_maps, axes=(-2, -1))
            k_maps = np.delete(k_maps, dead_lines, axis=-1)
            s_maps = sp.ifft(k_maps, axes=(-2, -1))

        # Store GT data without zero lines
        gt_nonzero_ksp = np.copy(k_image)

        # What is the readout direction?
        sampling_axis = -1 if self.direction == "y" else -2

        # Get locations of center and non-center
        if self.use_acs:
            # Fixed percentage of central lines, as per fastMRI
            if self.downsample >= 2 and self.downsample <= 6:
                num_central_slices = np.round(0.08 * k_image.shape[sampling_axis])
            else:
                num_central_slices = np.round(0.04 * k_image.shape[sampling_axis])
            center_slice_idx = np.arange(
                (k_image.shape[sampling_axis] - num_central_slices) // 2,
                (k_image.shape[sampling_axis] + num_central_slices) // 2,
            )
        else:
            # A fixed number of central lines given by 'acs_lines'
            center_slice_idx = np.arange(
                (k_image.shape[sampling_axis] - self.acs_lines) // 2,
                (k_image.shape[sampling_axis] + self.acs_lines) // 2,
            )

        # Downsampling
        # !!! Unlike fastMRI, we always pick lines to ensure R = downsample
        if self.downsample > 1.01:
            # Candidates outside the central region
            random_slice_candidates = np.setdiff1d(
                np.arange(k_image.shape[sampling_axis]), center_slice_idx
            )

            # If masks are not fed, generate them on the spot
            if self.saved_masks is None:
                # Pick random lines outside the center location
                random_slice_idx = np.random.choice(
                    random_slice_candidates,
                    size=(int(k_image.shape[sampling_axis] // self.downsample) - len(center_slice_idx)),
                    replace=False,
                )

                # Create sampling mask and downsampled k-space data
                k_sampling_mask = np.isin(
                    np.arange(k_image.shape[sampling_axis]), np.hstack((center_slice_idx, random_slice_idx))
                )
            else:
                # Use the corresponding mask
                k_sampling_mask = self.saved_masks[idx]

            # Apply by deletion
            if self.direction == "y":
                k_image[..., np.logical_not(k_sampling_mask)] = 0.0
                k_sampling_mask = k_sampling_mask[None, ...]
            elif self.direction == "x":
                k_image[..., np.logical_not(k_sampling_mask), :] = 0.0
                k_sampling_mask = k_sampling_mask[..., None]
        else:
            # No downsampling, all ones
            k_sampling_mask = np.ones((1, k_image.shape[-1]))

        # Get ACS region
        if self.use_acs:
            if self.direction == "y":
                acs = k_image[..., center_slice_idx.astype(np.int)]
            elif self.direction == "x":
                acs = k_image[..., center_slice_idx.astype(np.int), :]
        else:
            # Scale w.r.t. to the entire image
            acs = gt_ksp

        # Normalize k-space based on ACS
        max_acs = np.max(np.abs(acs))
        k_normalized_image = k_image / max_acs
        gt_ksp = gt_ksp / max_acs
        # Scaled GT RSS
        ref_rss = ref_rss / max_acs
        data_range = np.max(ref_rss)

        # Initial sensitivity maps
        x_coils = sp.ifft(k_image, axes=(-2, -1))
        x_rss = np.linalg.norm(x_coils, axis=0, keepdims=True)
        init_maps = sp.resize(sp.fft(x_coils / x_rss, axes=(-2, -1)), oshape=self.mps_kernel_shape)

        # Complex-to-real kspace
        k_normalized_image = np.stack((np.real(k_normalized_image), np.imag(k_normalized_image)), axis=-1)

        sample = {
            "idx": idx,
            "ksp": k_normalized_image.astype(np.float32),
            "gt_ksp": gt_ksp.astype(np.complex64),
            "gt_nonzero_ksp": gt_nonzero_ksp.astype(np.complex64),
            "s_maps": np.stack((np.real(s_maps), np.imag(s_maps)), axis=-1).astype(np.float32),
            "s_maps_cplx": s_maps.astype(np.complex64),
            "s_maps_full": s_maps_full.astype(np.complex64),
            "init_maps": init_maps.astype(np.complex64),
            "mask": k_sampling_mask.astype(np.float32),
            "acs_lines": len(center_slice_idx),
            "dead_lines": dead_lines,
            "ref_rss": ref_rss.astype(np.float32),
            "data_range": data_range,
            "core_file": core_file,
            "core_slice": core_slice,
            "max_acs": max_acs,
        }

        return sample
