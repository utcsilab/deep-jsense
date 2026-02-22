#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def ifft(x):
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.ifftshift(x, dim=(-2, -1))

    return x


def fft(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))

    return x


def itemize(x):
    """Converts a Tensor into a list of Python numbers."""
    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()


# Complex dot product of two complex-valued multidimensional Tensors
def zdot_batch(x1, x2):
    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1) * x2, (batch, -1)).sum(1)


# Same, applied to self --> squared L2-norm
def zdot_single_batch(x):
    return zdot_batch(x, x)
