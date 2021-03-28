#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Self implemented
class MCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.mean(torch.mean(torch.sum(
            torch.square(torch.abs(X - Y)), dim=(-1, -2)),
                dim=-1))
        
        return error_norm

# Self implemented
class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_mse = torch.mean(torch.square(X - Y))
        self_max  = torch.max(torch.square(X))
        
        return error_mse / self_max

# Self implemented
class NMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.square(torch.norm(X - Y))
        self_norm  = torch.square(torch.norm(X))
        
        return error_norm / self_norm

# From fMRI
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

# Custom step scheduler
def lambda_step(drop_epochs, drop_factor):
    # Inner function trick
    def core_function(epoch):
        if np.isin(epoch, drop_epochs):
            return drop_factor
        else:
            return 1.
        
    return core_function