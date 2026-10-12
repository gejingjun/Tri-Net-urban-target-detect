"""SDFE — Spatial-Differential Feature Enhancement (paper Sec. 3.2).

Mounting point: post-backbone, pre-fusion; applied to (X_vis, X_ir) at each
backbone stage l. Implements Eqs. 2-8:

    Eq. 2   X_diff   = |X_vis - X_ir|
    Eq. 3   A_spatial = sigmoid(Conv3x3(DWConv5x5(X_diff)))
    Eq. 4-5 local mean/std via AvgPool7 / StdPool7
    Eq. 6   X_norm   = (X_diff - mu_local) / (sigma_local + eps)
    Eq. 7   X_hat    = (1/sqrt(C)) * L2(alpha * X_norm)
    Eq. 8a  F_vis = X_vis + A_spatial * X_vis * tanh(gamma * X_hat + beta)
    Eq. 8b  F_ir  = X_ir  + A_spatial * X_ir  * tanh(gamma * X_hat + beta)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv


class StdPool2d(nn.Module):
    """StdPool_{k x k}: local standard deviation via difference of means
    (E[y] and [y^2]) — Eq. 5."""

    def __init__(self, k=7):
        super().__init__()
        self.avg = nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False)
        self.k2 = k * k

    def forward(self, x):
        mean = self.avg(x)
        mean_sq = self.avg(x * x)
        var = (mean_sq - mean * mean).clamp(min=0.0)
        return torch.sqrt(var + 1e-12)


class SDFE(nn.Module):
    """Spatial-Differential Feature Enhancement (Sec. 3.2, Eqs. 2-8)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Eq. 3: DWConv5x5 -> Conv3x3 -> 1 channel -> sigmoid
        self.dw = Conv(channels, channels, k=5, s=1, g=channels)
        self.attn = nn.Sequential(Conv(channels, 1, k=3, act=False), nn.Sigmoid())
        # Eq. 4-6: local normalization (parameter-free pooling path)
        self.mean_pool = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)
        self.std_pool = StdPool2d(7)
        # Eq. 7: learnable scaling alpha (per-channel) and L2 normalisation
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        # Eq. 8: learnable tanh gating gamma / beta (per-channel)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor):
        # Eq. 2
        x_diff = (x_vis - x_ir).abs()
        # Eq. 3
        a_spatial = self.attn(self.dw(x_diff))            # (B, 1, H, W) in [0,1]
        # Eq. 4-6
        mu = self.mean_pool(x_diff)
        sigma = self.std_pool(x_diff)
        x_norm = (x_diff - mu) / (sigma + self.eps)
        # Eq. 7:  X_hat = (1/sqrt(C)) * L2(alpha * X_norm), L2 over channels
        C = x_norm.shape[1]
        xn = self.alpha * x_norm
        x_hat = xn / xn.pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-6) / (C ** 0.5)
        # Eq. 8
        gate = torch.tanh(self.gamma * x_hat + self.beta)
        f_vis = x_vis + a_spatial * x_vis * gate
        f_ir = x_ir + a_spatial * x_ir * gate
        return f_vis, f_ir
