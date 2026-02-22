#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:17:34 2020

@author: yanni
"""

import torch
import sigpy as sp
import numpy as np
import copy as copy

from core_ops import TorchHybridSense, TorchHybridImage
from core_ops import TorchMoDLSense, TorchMoDLImage
from utils import fft, ifft

from opt import ZConjGrad
from resnet import ResNet


# Unrolled J-Sense in MoDL style
class MoDLDoubleUnroll(torch.nn.Module):
    def __init__(self, hparams):
        super(MoDLDoubleUnroll, self).__init__()
        # Storage
        self.verbose = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block1_max_iter = hparams.block1_max_iter
        self.block2_max_iter = hparams.block2_max_iter
        self.cg_eps = hparams.cg_eps

        # Modes
        self.mode = hparams.mode
        self.use_img_net = hparams.use_img_net
        self.use_map_net = hparams.use_map_net

        # Initial variables
        self.map_init = hparams.map_init
        self.img_init = hparams.img_init
        # Logging
        self.logging = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks = hparams.img_blocks
        self.img_sep = hparams.img_sep
        # Attention parameters
        self.att_config = hparams.att_config

        # Size parameters
        self.mps_kernel_shape = hparams.mps_kernel_shape  # B x C x h x w
        # Get useful values
        self.num_coils = self.mps_kernel_shape[-3]
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block1_l2lam = torch.nn.Parameter(torch.tensor(hparams.l2lam_init * np.ones((1))).cuda())
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(hparams.l2lam_init * np.ones((1))).cuda())
        else:
            self.block1_l2lam = torch.tensor(hparams.l2lam_init * np.ones((1))).cuda()
            self.block2_l2lam = torch.tensor(hparams.l2lam_init * np.ones((1))).cuda()

        # Initialize image module
        if hparams.use_img_net:
            # Initialize ResNet module
            if self.img_sep:  # Do we use separate networks at each unroll?
                self.image_net = torch.nn.ModuleList(
                    hparams.meta_unrolls_end
                    * [
                        ResNet(
                            in_channels=2,
                            latent_channels=self.img_channels,
                            num_blocks=self.img_blocks,
                            kernel_size=3,
                            batch_norm=False,
                        )
                    ]
                )
            else:
                self.image_net = ResNet(
                    in_channels=2,
                    latent_channels=self.img_channels,
                    num_blocks=self.img_blocks,
                    kernel_size=3,
                    batch_norm=False,
                )
        else:
            # Bypass
            self.image_net = torch.nn.Identity()

        # Initialize map module
        if hparams.use_map_net:
            # Intialize ResNet module
            self.maps_net = ResNet(
                in_channels=2,
                latent_channels=self.img_channels,
                num_blocks=self.img_blocks,
                kernel_size=3,
                batch_norm=False,
            )
        else:
            # Bypass
            self.maps_net = torch.nn.Identity()

        # Initial 'fixed' maps
        # See in 'forward' for the exact initialization depending on mode
        self.init_maps_kernel = (
            0.0 * torch.randn((self.batch_size,) + tuple(self.mps_kernel_shape) + (2,)).cuda()
        )
        self.init_maps_kernel = torch.view_as_complex(self.init_maps_kernel)

    # Get torch operators for the entire batch
    def get_core_torch_ops(self, mps_kernel, img_kernel, mask, direction):
        # List of output ops
        normal_ops, adjoint_ops, forward_ops = [], [], []

        # For each sample in batch
        for idx in range(self.batch_size):
            if self.mode == "DeepJSense":
                # Type
                if direction == "ConvSense":
                    forward_op, adjoint_op, normal_op = TorchHybridSense(
                        self.img_kernel_shape,
                        mps_kernel[idx],
                        mask[idx],
                        self.img_conv_shape,
                        self.ksp_padding,
                        self.maps_padding,
                    )
                elif direction == "ConvImage":
                    forward_op, adjoint_op, normal_op = TorchHybridImage(
                        self.mps_kernel_shape,
                        img_kernel[idx],
                        mask[idx],
                        self.img_conv_shape,
                        self.ksp_padding,
                        self.maps_padding,
                    )
            elif self.mode == "MoDL":
                # Type
                if direction == "ConvSense":
                    forward_op, adjoint_op, normal_op = TorchMoDLSense(mps_kernel[idx], mask[idx])
                elif direction == "ConvImage":
                    forward_op, adjoint_op, normal_op = TorchMoDLImage(img_kernel[idx], mask[idx])

            # Add to lists
            normal_ops.append(normal_op)
            adjoint_ops.append(adjoint_op)
            forward_ops.append(forward_op)

        # Return operators
        return normal_ops, adjoint_ops, forward_ops

    # Given a batch of inputs and ops, get a single batch operator
    def get_batch_op(self, input_ops, batch_size):
        # Inner function trick
        def core_function(x):
            # Store in list
            output_list = []
            for idx in range(batch_size):
                output_list.append(input_ops[idx](x[idx])[None, ...])
            # Stack and return
            return torch.cat(output_list, dim=0)

        return core_function

    def forward(self, data, meta_unrolls=1):
        # Use the full accelerated k-space
        ksp = data["ksp"]
        mask = data["mask"]  # 2D mask (or whatever, easy to adjust)

        # Get image kernel shape - dynamic and includes padding
        if self.mode == "DeepJSense":
            self.img_kernel_shape = [
                ksp.shape[-3] + self.mps_kernel_shape[-2] - 1,
                ksp.shape[-2] + self.mps_kernel_shape[-1] - 1,
            ]  # H x W
            self.img_conv_shape = [
                self.num_coils,
                self.img_kernel_shape[-2] - self.mps_kernel_shape[-2] + 1,
                self.img_kernel_shape[-1] - self.mps_kernel_shape[-1] + 1,
            ]  # After convoluting with map kernel

            # Compute all required padding parameters
            self.padding = (
                self.img_conv_shape[-2] - 1,
                self.img_conv_shape[-1] - 1,
            )  # Outputs a small kernel

            # Decide based on the number of k-space lines
            if np.mod(ksp.shape[-2], 2) == 0:
                self.maps_padding = (
                    int(np.ceil(self.padding[-2] / 2)),
                    int(np.floor(self.padding[-2] / 2)),
                    int(np.ceil(self.padding[-1] / 2)),
                    int(np.floor(self.padding[-1] / 2)),
                )
                self.ksp_padding = (
                    int(np.ceil((self.img_kernel_shape[-2] - self.img_conv_shape[-2]) / 2)),
                    int(np.floor((self.img_kernel_shape[-2] - self.img_conv_shape[-2]) / 2)),
                    int(np.ceil((self.img_kernel_shape[-1] - self.img_conv_shape[-1]) / 2)),
                    int(np.floor((self.img_kernel_shape[-1] - self.img_conv_shape[-1]) / 2)),
                )
            else:
                # !!! Input ksp has to be of even shape
                assert False

        elif self.mode == "MoDL":  # No padding
            pass  # Nothing needed

        # Initializers
        with torch.no_grad():
            # View input as complex
            ksp = torch.view_as_complex(ksp)

            # Initial maps
            if self.map_init == "fixed":
                est_maps_kernel = self.init_maps_kernel
            elif self.map_init == "estimated":
                # From dataloader
                est_maps_kernel = data["init_maps"].type(torch.complex64)
            elif self.map_init == "espirit":
                # From dataloader
                est_maps_kernel = data["s_maps_cplx"]

            # Initial image
            if self.img_init == "fixed":
                est_img_kernel = sp.dirac(self.img_kernel_shape, dtype=np.complex64)[None, ...]
                est_img_kernel = np.repeat(est_img_kernel, self.batch_size, axis=0)
                # Image domain
                est_img_kernel = sp.ifft(est_img_kernel, axes=(-2, -1))
                est_img_kernel = torch.tensor(est_img_kernel, dtype=torch.cfloat).cuda()
            elif self.img_init == "estimated":
                # Get adjoint map operator
                _, adjoint_ops, _ = self.get_core_torch_ops(est_maps_kernel, None, mask, "ConvSense")
                adjoint_batch_op = self.get_batch_op(adjoint_ops, self.batch_size)
                # Apply
                est_img_kernel = adjoint_batch_op(ksp).type(torch.complex64)

        # Logging outputs
        if self.logging:
            # Kernels after denoiser modules
            mps_kernel_denoised = []
            img_kernel_denoised = []
            # Estimated logs
            mps_logs, img_logs = [], []
            ksp_logs = []
            mps_logs.append(copy.deepcopy(est_maps_kernel))
            img_logs.append(copy.deepcopy(est_img_kernel))
            # Internal logs
            before_maps, after_maps = [], []
            att_logs = []

        # For each outer unroll
        for meta_idx in range(meta_unrolls):
            ## !!! Block 1
            if self.block1_max_iter > 0:
                if self.mode == "MoDL":
                    assert False, "Shouldnt be here!"

                # Get operators for images --> maps using image kernel
                normal_ops, adjoint_ops, forward_ops = self.get_core_torch_ops(
                    None, est_img_kernel, mask, "ConvImage"
                )
                # Get joint batch operators for adjoint and normal
                normal_batch_op, adjoint_batch_op = (
                    self.get_batch_op(normal_ops, self.batch_size),
                    self.get_batch_op(adjoint_ops, self.batch_size),
                )

                # Compute RHS
                if meta_idx == 0:
                    rhs = adjoint_batch_op(ksp)
                else:
                    rhs = adjoint_batch_op(ksp) + self.block1_l2lam[0] * est_maps_kernel

                # Get unrolled CG op
                cg_op = ZConjGrad(
                    rhs,
                    normal_batch_op,
                    l2lam=self.block1_l2lam[0],
                    max_iter=self.block1_max_iter,
                    eps=self.cg_eps,
                    verbose=self.verbose,
                )
                # Run CG
                est_maps_kernel = cg_op(est_maps_kernel)
                # Log
                if self.logging:
                    mps_logs.append(copy.deepcopy(est_maps_kernel))

                # Pre-process
                if not self.use_map_net:
                    pass
                else:
                    # Transform map kernel to image space
                    est_maps_kernel = ifft(est_maps_kernel)
                    # Convert to real and treat as a set
                    est_maps_kernel = torch.view_as_real(est_maps_kernel)
                    est_maps_kernel = est_maps_kernel.permute(0, 1, -1, 2, 3)
                    # Absorb batch dimension
                    est_maps_kernel = est_maps_kernel[0]

                # Log right before
                if self.logging:
                    before_maps.append(est_maps_kernel.cpu().detach().numpy())

                # Apply denoising network
                if not self.use_map_net:
                    pass
                else:
                    est_maps_kernel = self.maps_net(est_maps_kernel)

                # Log right after
                if self.logging:
                    after_maps.append(est_maps_kernel.cpu().detach().numpy())

                # Post-process
                if not self.use_map_net:
                    pass
                else:
                    # Inject batch dimension and re-arrange
                    est_maps_kernel = est_maps_kernel[None, ...]
                    est_maps_kernel = est_maps_kernel.permute(0, 1, 3, 4, 2).contiguous()

                    # Convert back to frequency domain
                    est_maps_kernel = torch.view_as_complex(est_maps_kernel)
                    est_maps_kernel = fft(est_maps_kernel)

                # Log
                if self.logging:
                    mps_kernel_denoised.append(copy.deepcopy(est_maps_kernel))

            ## !!! Block 2
            # Get operators for maps --> images using map kernel
            normal_ops, adjoint_ops, forward_ops = self.get_core_torch_ops(
                est_maps_kernel, None, mask, "ConvSense"
            )
            # Get joint batch operators for adjoint and normal
            normal_batch_op, adjoint_batch_op = (
                self.get_batch_op(normal_ops, self.batch_size),
                self.get_batch_op(adjoint_ops, self.batch_size),
            )

            # Compute RHS
            if meta_idx == 0:
                rhs = adjoint_batch_op(ksp)
            else:
                rhs = adjoint_batch_op(ksp) + self.block2_l2lam[0] * est_img_kernel

            # Get unrolled CG op
            cg_op = ZConjGrad(
                rhs,
                normal_batch_op,
                l2lam=self.block2_l2lam[0],
                max_iter=self.block2_max_iter,
                eps=self.cg_eps,
                verbose=self.verbose,
            )
            # Run CG
            est_img_kernel = cg_op(est_img_kernel)
            # Log
            if self.logging:
                img_logs.append(est_img_kernel)

            # Convert to reals
            est_img_kernel = torch.view_as_real(est_img_kernel)

            # Apply image denoising network in image space
            if self.img_sep:
                est_img_kernel = (
                    self.image_net[meta_idx](est_img_kernel.permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )
            else:
                est_img_kernel = (
                    self.image_net(est_img_kernel.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
                )

            # Convert to complex
            est_img_kernel = torch.view_as_complex(est_img_kernel)

            # Log
            if self.logging:
                img_kernel_denoised.append(est_img_kernel)

                # For all unrolls, construct k-space
                if meta_idx < meta_unrolls - 1:
                    _, _, scratch_ops = self.get_core_torch_ops(
                        est_maps_kernel, None, self.ones_mask, "ConvSense"
                    )
                    scratch_batch_op = self.get_batch_op(scratch_ops, self.batch_size)
                    est_ksp = scratch_batch_op(est_img_kernel)
                    # Log
                    ksp_logs.append(est_ksp)

        # Compute output coils with an unmasked convolution operator
        _, _, forward_ops = self.get_core_torch_ops(est_maps_kernel, None, self.ones_mask, "ConvSense")
        forward_batch_op = self.get_batch_op(forward_ops, self.batch_size)
        est_ksp = forward_batch_op(est_img_kernel)

        if self.logging:
            # Add final ksp to logs
            ksp_logs.append(est_ksp)
            # Glue logs
            mps_logs = torch.cat(mps_logs, dim=0)
            img_logs = torch.cat(img_logs, dim=0)
            if not self.mode == "MoDL":
                mps_kernel_denoised = torch.cat(mps_kernel_denoised, dim=0)
            img_kernel_denoised = torch.cat(img_kernel_denoised, dim=0)

        if self.logging:
            return (
                est_img_kernel,
                est_maps_kernel,
                est_ksp,
                mps_logs,
                img_logs,
                mps_kernel_denoised,
                img_kernel_denoised,
                ksp_logs,
                before_maps,
                after_maps,
                att_logs,
            )
        else:
            return est_img_kernel, est_maps_kernel, est_ksp
