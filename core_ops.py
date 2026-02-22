#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.fft as torch_fft

from torch.nn import functional as F
from datagen import crop


# Sense in Image space, exactly like in MoDL
# img -> mult (broad) -> FFT -> mask -> ksp
# ksp -> mask -> IFFT -> mult (conj) -> sum (coils) -> img
def TorchMoDLSense(mps_kernel, mask):
    # Get image representation of padded maps kernel
    img_y = mps_kernel
    # Get masks with right sizes
    mask_fw_ext = mask[None, ...]
    mask_adj_ext = mask[None, ...]

    # Forward
    def forward_op(img_kernel):
        # Pointwise complex multiply with maps
        mult_result = img_kernel[None, ...] * img_y

        # Convert back to k-space
        result = torch_fft.ifftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm="ortho")
        result = torch_fft.fftshift(result, dim=(-2, -1))

        # Multiply with mask
        result = result * mask_fw_ext

        return result

    # Adjoint
    def adjoint_op(ksp):
        # Multiply input with mask and pad
        ksp_padded = ksp * mask_adj_ext

        # Get image representation of ksp
        img_ksp = torch_fft.fftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm="ortho")
        img_ksp = torch_fft.ifftshift(img_ksp, dim=(-2, -1))

        # Pointwise complex multiply with complex conjugate maps
        mult_result = img_ksp * torch.conj(img_y)

        # Sum on coil axis
        x_adj = torch.sum(mult_result, dim=0)

        return x_adj

    # Normal operator
    def normal_op(img_kernel):
        return adjoint_op(forward_op(img_kernel))

    return forward_op, adjoint_op, normal_op


# Image in image space, exactly like in MoDL
# maps -> mult (broad) -> FFT -> mask -> ksp
# ksp -> mask -> IFFT -> mult (conj broad) -> maps
def TorchMoDLImage(img_kernel, mask):
    # Get image representation of image kernel
    img_x = img_kernel
    # Get masks with right sizes
    mask_fw_ext = mask[None, ...]
    mask_adj_ext = mask[None, ...]

    # Forward operator
    def forward_op(mps_kernel):
        # Pointwise complex multiply with maps
        mult_result = mps_kernel * img_x

        # Convert back to k-space
        result = torch_fft.ifftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm="ortho")
        result = torch_fft.fftshift(result, dim=(-2, -1))

        # Multiply with mask
        result = result * mask_fw_ext

        return result

    # Adjoint operator
    def adjoint_op(ksp):
        # Multiply input with mask and pad
        ksp_padded = ksp * mask_adj_ext

        # Get image representations
        img_ksp = torch_fft.fftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm="ortho")
        img_ksp = torch_fft.ifftshift(img_ksp, dim=(-2, -1))

        # Pointwise complex multiply (with conjugate image and broadcasting)
        mult_result = img_ksp * torch.conj(img_x)[None, ...]

        # Central crop
        y_adj = mult_result

        return y_adj

    # Normal operator
    def normal_op(mps_kernel):
        return adjoint_op(forward_op(mps_kernel))

    return forward_op, adjoint_op, normal_op


# Our 'hybrid' implementation
# ConvSense Forward - we're given an image kernel in image space, and we output k-space
# mask <- FFT <- pointwise mult with map kernel in image space <- image kernel in image space
# ConvSense Adjoint - we're given k-space, and we output an image kernel in image space
# mask -> IFFT -> pointwise mult with conj map kernel in image space -> estimated image kernel in image space
def TorchHybridSense(img_kernel_shape, mps_kernel, mask, img_full_shape, ksp_padding, maps_padding):
    # Get image representation of padded maps kernel
    y_padded = F.pad(mps_kernel, (maps_padding[-2], maps_padding[-1], maps_padding[-4], maps_padding[-3]))
    # This only happens once
    img_y = torch_fft.fftshift(y_padded, dim=(-2, -1))
    img_y = torch_fft.ifft2(img_y, dim=(-2, -1), norm="ortho")
    img_y = torch_fft.ifftshift(img_y, dim=(-2, -1))
    # Get masks with right sizes
    mask_fw_ext = mask[None, ...]
    mask_adj_ext = mask[None, None, ...]

    # Forward
    def forward_op(img_kernel):
        # Pointwise complex multiply with maps
        mult_result = img_kernel[None, ...] * img_y

        # Convert back to k-space
        # !!! The squared normalization before cancels the required normalization here
        result = torch_fft.ifftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm="ortho")
        result = torch_fft.fftshift(result, dim=(-2, -1))

        # Central crop
        result = crop(result, img_full_shape[-2], img_full_shape[-1])
        # Multiply with mask
        result = result * mask_fw_ext

        return result

    # Adjoint
    def adjoint_op(ksp):
        # Multiply input with mask and pad
        ksp_padded = F.pad(
            ksp * mask_adj_ext[0], (ksp_padding[-2], ksp_padding[-1], ksp_padding[-4], ksp_padding[-3])
        )

        # Get image representation of ksp
        img_ksp = torch_fft.fftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm="ortho")
        img_ksp = torch_fft.ifftshift(img_ksp, dim=(-2, -1))

        # Pointwise complex multiply with complex conjugate maps
        mult_result = img_ksp * torch.conj(img_y)
        # Sum on coil axis
        x_adj = torch.sum(mult_result, dim=0)

        return x_adj

    # Normal operator
    def normal_op(img_kernel):
        return adjoint_op(forward_op(img_kernel))

    return forward_op, adjoint_op, normal_op


# ConvImage Forward - we input a map kernel in k-space, and we output k-space
# mask <- FFT <- pointwise mult with image kernel in image space <- IFFT <- pad with zeroes <- map kernel in k-space
# map kernel in k-space -> pad with zeroes -> IFFT ('ortho') -> pointwise mult with image kernel in image space -> FFT ('backward') -> crop -> mask -> coil images in k-space
# ConvImage Adjoint - we input k-space, and we output a map kernel in k-space
# coil images in k-space -> mask -> pad with zeroes -> IFFT ('ortho') -> pointwise mult with conj image kernel -> FFT ('backward') -> crop -> map kernel in k-space
def TorchHybridImage(mps_kernel_shape, img_kernel, mask, img_full_shape, ksp_padding, maps_padding):
    # Get image representation of image kernel
    img_x = img_kernel
    # Get masks with right sizes
    mask_fw_ext = mask[None, ...]
    mask_adj_ext = mask[None, None, ...]

    # Forward operator
    def forward_op(mps_kernel):
        # Get image representation of padded maps kernel
        y_padded = F.pad(
            mps_kernel, (maps_padding[-2], maps_padding[-1], maps_padding[-4], maps_padding[-3])
        )
        img_y = torch_fft.ifftshift(y_padded, dim=(-2, -1))
        img_y = torch_fft.ifft2(img_y, dim=(-2, -1), norm="ortho")
        img_y = torch_fft.fftshift(img_y, dim=(-2, -1))

        # Pointwise complex multiply with image kernel
        mult_result = img_y * img_kernel

        # Convert back to k-space
        # !!! The squared normalization cancels the required normalization here
        result = torch_fft.fftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm="backward")
        result = torch_fft.ifftshift(result, dim=(-2, -1))

        # Central crop
        result = crop(result, img_full_shape[-2], img_full_shape[-1])
        # Multiply with mask
        result = result * mask_fw_ext

        return result

    # Adjoint operator
    def adjoint_op(ksp):
        # Multiply input with mask and pad
        ksp_padded = F.pad(
            ksp * mask_adj_ext[0], (ksp_padding[-2], ksp_padding[-1], ksp_padding[-4], ksp_padding[-3])
        )
        # Get image representations
        img_ksp = torch_fft.ifftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm="ortho")
        img_ksp = torch_fft.fftshift(img_ksp, dim=(-2, -1))

        # Pointwise complex multiply (with conjugate image and broadcasting)
        mult_result = img_ksp * torch.conj(img_x)[None, ...]

        # Convert back to k-space
        result = torch_fft.fftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm="backward")
        result = torch_fft.ifftshift(result, dim=(-2, -1))

        # Central crop
        y_adj = crop(result, mps_kernel_shape[-2], mps_kernel_shape[-1])

        return y_adj

    # Normal operator
    def normal_op(mps_kernel):
        return adjoint_op(forward_op(mps_kernel))

    return forward_op, adjoint_op, normal_op
