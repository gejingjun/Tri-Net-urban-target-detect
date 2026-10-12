"""VADE — Validity-Aware Depth Encoding and Fusion (paper Sec. 3.3).

Phase A (Sec. 3.3.1, Eqs. 9-14): deterministic, zero-parameter depth encoding
applied to the raw metric depth map *before* the backbone:

    Eq. 9   D_valid = 1[D > 0]
    Eq. 10  D_inv   = 1 / clip(D, d_min, d_max) * d_max
    Eq. 11  tau_k   = Q_{p_k}(D | D > 0)            (quantile thresholds)
    Eq. 12  D_bin_k = 1[tau_{k-1} < D <= tau_k] * (D - tau_{k-1})/(tau_k - tau_{k-1})
    Eq. 13  E_dep   = [D_valid, D_inv, D_bin_1 ... D_bin_{Cd-2}]  (Cd=6)
    Eq. 14  routing: metric pipeline if D_max > 255 else pseudo [D/255, 1[D>0]]

Phase B (Sec. 3.3.2, Eqs. 15-17): learned validity gating at the fusion stage:

    Eq. 15  g_pixel  = sigmoid(Conv1x1(E_dep))
    Eq. 16  g_region = AvgPool_{k x k}(g_pixel), k = 16
    Eq. 17  X_dep~   = (g_pixel * g_region) * X_dep

Modality dropout (Sec. 3.3.2d): during training, E_dep <- 0 with p_drop=0.15.
Auxiliary loss L_valid = BCE(g_pixel, D_valid) (Sec. 3.6).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults measured on UrbanTraffic-3M (paper Sec. 3.3.1)
D_MIN_MM = 300.0
D_MAX_MM = 19999.0
C_D = 6
QUANTILES = (0.25, 0.50, 0.75)
# tau = Q_p over the valid-depth distribution; paper measured values:
TAU_PAPER = (4702.0, 7556.0, 10848.0)


class DepthEncoder:
    """VADE Phase A — zero-parameter range-aligned encoding (Eqs. 9-14).

    Call ``fit`` once on the training split's valid depth pixels to compute
    quantile thresholds (Eq. 11), then ``encode`` per image. Thresholds are
    persisted in the dataset stats JSON so evaluation reproduces training.

    mode (paper Table 8 — Depth Encoding Comparison):
      'vade_a'     : full Phase A, Cd=6 quantile bins + validity + inverse depth
      'raw'        : single channel, raw depth / 1000
      'near_mid_far' : 3 fixed bins (edges from ``fixed_edges``, e.g.
                       [10000, 30000] for the (10m,30m) row or [5000, 11000])
      'inv_valid'  : 2 channels: inverse depth + validity indicator
    """

    def __init__(self, c_d=C_D, d_min=D_MIN_MM, d_max=D_MAX_MM,
                 quantiles=QUANTILES, mode="vade_a", fixed_edges=None):
        self.c_d = c_d if mode == "vade_a" else {"raw": 1, "near_mid_far": 3,
                                                 "inv_valid": 2}[mode]
        self.mode = mode
        self.d_min = d_min
        self.d_max = d_max
        self.quantiles = quantiles
        self.fixed_edges = list(fixed_edges) if fixed_edges else [10000.0, 30000.0]
        self.tau = list(TAU_PAPER)      # fallback = paper-measured thresholds

    # ---- Eq. 11 -----------------------------------------------------------
    def fit(self, valid_depths: torch.Tensor) -> "DepthEncoder":
        """valid_depths: 1-D tensor of positive depth values (mm)."""
        qs = torch.tensor(self.quantiles, dtype=valid_depths.dtype)
        self.tau = torch.quantile(valid_depths.float(), qs).tolist()
        return self

    # ---- Eq. 13 -----------------------------------------------------------
    def encode(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (H, W) float32 in millimetres (0 = invalid).
        Returns (c_d, H, W) float32."""
        d = depth.float()
        # Eq. 14 deterministic routing for uint8 pseudo-depth (non-metric)
        if float(d.max()) <= 255.0 and self.mode in ("vade_a", "inv_valid"):
            pseudo = d / 255.0
            valid = (d > 0).float()
            out = torch.zeros(self.c_d, *d.shape, dtype=torch.float32)
            out[0] = pseudo
            out[1] = valid
            return out
        valid = (d > 0).float()                                  # Eq. 9
        clipped = d.clamp(min=self.d_min, max=self.d_max)
        d_inv = (1.0 / clipped) * self.d_max                    # Eq. 10
        if self.mode == "raw":
            return (d / 1000.0)[None]
        if self.mode == "inv_valid":
            return torch.stack([d_inv / self.d_max, valid], 0)
        if self.mode == "near_mid_far":
            edges = [0.0] + list(self.fixed_edges) + [float("inf")]
            chans = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                inb = ((d > lo) & (d <= hi)).float()
                hi_c = min(hi, 20000.0)
                frac = ((d - lo) / max(hi_c - lo, 1e-6)).clamp(0, 1) * inb
                chans.append(frac)
            return torch.stack(chans, 0)
        edges = [self.d_min] + list(self.tau) + [self.d_max]    # Eq. 11/12
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = ((d > lo) & (d <= hi)).float()
            frac = ((d - lo) / max(hi - lo, 1e-6)).clamp(0, 1) * in_bin  # Eq. 12
            bins.append(frac)
        # Eq. 13: [valid, inv_depth, bin_1 .. bin_{Cd-2}]
        out = torch.stack([valid, d_inv] + bins, dim=0)
        return out

    def state_dict(self):
        return {"tau": list(self.tau)}

    def load_state_dict(self, sd):
        self.tau = list(sd["tau"])
        return self


class DepthEncoderBatch(nn.Module):
    """Batched wrapper of :class:`DepthEncoder` (keeps gradients flowing
    through nothing — Phase A is deterministic, no parameters)."""

    def __init__(self, encoder: DepthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, depth_b1hw: torch.Tensor) -> torch.Tensor:
        outs = [self.encoder.encode(depth_b1hw[b, 0]) for b in range(depth_b1hw.shape[0])]
        return torch.stack(outs, 0)


class VADEGate(nn.Module):
    """VADE Phase B — validity-gated depth fusion (Eqs. 15-17) + dropout (§3.3.2d)."""

    def __init__(self, c_d=C_D, region_k=16, p_drop=0.15):
        super().__init__()
        # Eq. 15: Conv1x1(E_dep) -> sigmoid, initialised to approximate D_valid
        self.proj = nn.Conv2d(c_d, 1, 1)
        nn.init.zeros_(self.proj.bias)
        w = torch.zeros(1, c_d, 1, 1)
        w[0, 0] = 4.0          # channel 0 == validity indicator (Eq. 9)
        self.proj.weight.data.copy_(w)
        self.region_k = region_k
        self.p_drop = p_drop
        self.last_g_pixel: torch.Tensor | None = None  # for L_valid supervision
        self.last_target: torch.Tensor | None = None   # D_valid downsampled

    def forward(self, x_dep: torch.Tensor, e_dep: torch.Tensor) -> torch.Tensor:
        """x_dep (B,C,H,W) depth-branch features; e_dep (B,Cd,Hs,Ws) encoded map.
        e_dep is bilinearly resized to the feature resolution (shared gate
        applied at each pyramid level P3/P4/P5, Sec. 3.3.2 mounting points)."""
        if e_dep.shape[-2:] != x_dep.shape[-2:]:
            e = F.interpolate(e_dep, size=x_dep.shape[-2:], mode="bilinear",
                              align_corners=False)
        else:
            e = e_dep
        # Modality dropout (§3.3.2d): E_dep <- 0 AND the depth contribution is
        # removed for the dropped samples, forcing an appearance-only fallback
        drop = None
        if self.training and self.p_drop > 0:
            drop = torch.rand(x_dep.shape[0], 1, 1, 1, device=e.device) < self.p_drop
            e = torch.where(drop, torch.zeros_like(e), e)
        # Eq. 15
        g_pixel = torch.sigmoid(self.proj(e))
        # Eq. 16 (keep gate in [0,1]); resize back in case padding shifts the grid
        g_region = F.avg_pool2d(g_pixel, self.region_k, stride=1,
                                padding=self.region_k // 2,
                                count_include_pad=False)
        if g_region.shape[-2:] != g_pixel.shape[-2:]:
            g_region = F.interpolate(g_region, size=g_pixel.shape[-2:],
                                     mode="bilinear", align_corners=False)
        # Eq. 17
        x_gated = (g_pixel * g_region) * x_dep
        if drop is not None:
            x_gated = torch.where(drop, torch.zeros_like(x_gated), x_gated)
        self.last_g_pixel = g_pixel
        self.last_target = (e[:, :1] > 0.5).float() if e.shape[1] >= 1 else None
        return x_gated
