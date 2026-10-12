"""MS-GDFF — Multi-Scale Guided Feature Fusion (paper Sec. 3.4).

Scale-aware tri-modal routing on the VADE-gated features. Implements
Eqs. 18-22:

    Eq. 18  f_g      = GAP([F_vis; F_ir; X_dep~])                    in R^{3C}
    Eq. 19  [w_s,w_m,w_l] = softmax(MLP(f_g))
    Eq. 20  P_small  = C3k2(C)
    Eq. 21  P_medium = RCCA(C3k2(C));      P_large = RCCA(RCCA(C3k2(C)))
    Eq. 22  F_shared = w_s*P_s + w_m*P_m + w_l*P_l
          F_fuse     = CBS(sum_k w_k * P_k)

Also hosts the tri-modal quality gating unit (QGU, Eq. 23):
    G_m = QGU(F_m; k_DW, k_dil), m in {vis, ir, dep}
with scale-adaptive receptive fields (kernel config follows the dominant
scale predicted by the router).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import C3k2, CBS, ChannelAttention, RCCA


class ScaleRouter(nn.Module):
    """GAP -> MLP -> softmax over {small, medium, large} paths (Eqs. 18-19)."""

    def __init__(self, c3_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c3_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3))

    def forward(self, cat_feat: torch.Tensor):
        g = cat_feat.mean((2, 3))                     # Eq. 18
        w = torch.softmax(self.mlp(g), dim=-1)        # Eq. 19
        return w                                      # (B, 3)


class QGU(nn.Module):
    """Quality Gating Unit extended to tri-modal inputs (Eq. 23).

    A lightweight conv stack producing a per-pixel quality map G_m in [0,1];
    kernel configuration adapts to the dominant scale:
      small  -> 3x3 depthwise + 3x3 dilated
      medium -> 5x5 depthwise + 3x3 dilated
      large  -> 7x7 depthwise + 5x5 dilated
    """

    CFG = {"small": (3, 3), "medium": (5, 3), "large": (7, 5)}

    def __init__(self, c, scale="medium"):
        super().__init__()
        k_dw, k_dil = self.CFG[scale]
        dilation = 2 if k_dil >= 5 else 1
        pad = (k_dil - 1) // 2 * dilation
        self.dw = nn.Conv2d(c, c, k_dw, padding=k_dw // 2, groups=c)
        self.dil = nn.Conv2d(c, 1, k_dil, padding=pad, dilation=dilation)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.dil(torch.relu(self.dw(x))))     # (B,1,H,W)


class MSGDFF(nn.Module):
    """Scale-aware tri-modal fusion (Sec. 3.4).

    Returns F_fuse plus the QGU quality maps {G_vis, G_ir, G_dep} and
    F_shared consumed by RU-Gate (Sec. 3.5).
    """

    def __init__(self, c_in: int, c_out: int, n_bottleneck: int = 2,
                 rcca_reduction: int = 4):
        super().__init__()
        self.cat_ch = 3 * c_in
        # Eq. 20/21 paths
        self.path_small = C3k2(self.cat_ch, c_out, n=n_bottleneck, shortcut=True)
        self.path_mid_a = C3k2(self.cat_ch, c_out, n=n_bottleneck, shortcut=True)
        self.rcca_m = RCCA(c_out, c_out, reduction=rcca_reduction, recurrent=1)
        self.path_big_a = C3k2(self.cat_ch, c_out, n=n_bottleneck, shortcut=True)
        self.rcca_b1 = RCCA(c_out, c_out, reduction=rcca_reduction, recurrent=1)
        self.rcca_b2 = RCCA(c_out, c_out, reduction=rcca_reduction, recurrent=1)
        self.router = ScaleRouter(self.cat_ch)
        self.cbs = CBS(c_out, c_out, 1)               # final CBS of Eq. 22
        self.cam = ChannelAttention(c_out)            # channel modulation (Sec. 3.4)
        # scale-adaptive QGUs (Eq. 23), one per modality
        self.qgu_vis = QGU(c_in, "small")
        self.qgu_ir = QGU(c_in, "medium")
        self.qgu_dep = QGU(c_in, "large")

    def forward(self, f_vis, f_ir, f_dep_gated):
        c = torch.cat([f_vis, f_ir, f_dep_gated], dim=1)
        w = self.router(c)                                    # Eq. 18-19 (B,3)
        p_s = self.path_small(c)                              # Eq. 20
        p_m = self.rcca_m(self.path_mid_a(c))                 # Eq. 21
        p_l = self.rcca_b2(self.rcca_b1(self.path_big_a(c)))  # Eq. 21
        ws = w[:, :, None, None]
        f_shared = ws[:, 0:1] * p_s + ws[:, 1:2] * p_m + ws[:, 2:3] * p_l   # Eq. 22
        f_fuse = self.cbs(self.cam(f_shared))
        g_vis = self.qgu_vis(f_vis)
        g_ir = self.qgu_ir(f_ir)
        g_dep = self.qgu_dep(f_dep_gated)
        return dict(f_fuse=f_fuse, f_shared=f_shared, w=w,
                    g_vis=g_vis, g_ir=g_ir, g_dep=g_dep)
