"""Shared building blocks used by every TFG-Net module.

Self-contained (no ultralytics internals required) implementations of the
components referenced by the paper:

* CBS           -- Conv + BatchNorm + SiLU (YOLOv8 basic block)
* DWConv / DWConv5x5 (Eq. 3), Conv3x3
* C3k2          -- cross-stage partial bottleneck from YOLO11 (Eq. 20)
* SPPF          -- spatial pyramid pooling fast
* RCCA          -- Recurrent Criss-Cross Attention (CCNet, Eq. 20-21)
* CAM           -- channel attention (Eq. in MS-GDFF description, Sec. 3.4)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Conv2d):
    """Conv + BN + SiLU (CBS)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, k // 2 if p is None else p,
                         groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(super().forward(x)))


# Conv + BatchNorm + SiLU == the paper's "CBS" block
CBS = Conv


class DWConv(Conv):
    """Depthwise conv (groups=c)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math_gcd(c1, c2), d=d, act=act)


def math_gcd(a, b):
    import math
    return math.gcd(int(a), int(b)) or 1


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k1=3, k2=3, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k1)
        self.cv2 = Conv(c_, c2, k2, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """C3k2 from YOLO11 (Eq. 20): CSP block with k×k bottlenecks.

    c3k=False -> plain conv bottlenecks (used by the YOLOv8 backbone);
    c3k=True  -> 5x5 depthwise-separable bottlenecks (YOLO11 style).
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, c3k=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        kb = 5 if c3k else k
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g=1 if c3k else g,
                       k1=kb if not c3k else 3, k2=kb)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class RCCA(nn.Module):
    """Recurrent Criss-Cross Attention (CCNet) — Eq. 20/21 path depth.

    Single criss-cross attention layer: every pixel gathers context along its
    full row and column (4·HW neighbours). `recurrent` iterations turn the
    2-hop criss-cross field into (almost) a global receptive field, which is
    exactly what CCNet uses; MS-GDFF stacks 1 or 2 RCCA blocks on top of the
    C3k2 path to build medium/large receptive-field fusion paths.
    """

    def __init__(self, c1, c2=None, reduction=4, recurrent=1):
        super().__init__()
        c2 = c2 or c1
        self.recurrent = recurrent
        cq = c1 // reduction
        self.key = Conv(c1, cq, 1)
        self.value = Conv(c1, c2, 1)
        self.q = Conv(c1, cq, 1)
        self.project = nn.Sequential(Conv(c2, c2, 1), nn.Sigmoid())

    @staticmethod
    def _criss_attend(q, k, v, h, w):
        """Attend every position to its whole row + whole column.

        q,k: (B, Ck, H*W), v: (B, C2, H*W). Returns gathered (B, C2, H*W).
        Equivalent to CCNet's masked softmax over 4*HW neighbours, computed
        efficiently as two independent softmax normalisations (row-wise and
        column-wise) whose results are averaged.
        """
        b, ck, hw = q.shape
        c2 = v.shape[1]
        # column direction: positions sharing the same row -> reshape (B*H, W, .)
        qr = q.view(b, ck, h, w).permute(0, 2, 1, 3).reshape(b * h, ck, w)
        kr = k.view(b, ck, h, w).permute(0, 2, 1, 3).reshape(b * h, ck, w)
        vr = v.view(b, c2, h, w).permute(0, 2, 1, 3).reshape(b * h, c2, w)
        ar = torch.softmax(torch.einsum("bcl,bcn->bln", qr, kr) / (ck ** 0.5), -1)
        gr = torch.einsum("bln,bcn->bcl", ar, vr)          # (B*H, C2, W)
        # row direction: positions sharing the same column -> (B*W, H, .)
        qc = q.view(b, ck, h, w).permute(0, 3, 1, 2).reshape(b * w, ck, h)
        kc = k.view(b, ck, h, w).permute(0, 3, 1, 2).reshape(b * w, ck, h)
        vc = v.view(b, c2, h, w).permute(0, 3, 1, 2).reshape(b * w, c2, h)
        ac = torch.softmax(torch.einsum("bcl,bcn->bln", qc, kc) / (ck ** 0.5), -1)
        gc = torch.einsum("bln,bcn->bcl", ac, vc)          # (B*W, C2, H)
        gr = gr.view(b, h, c2, w).permute(0, 2, 1, 3)      # (B, C2, H, W)
        gc = gc.view(b, w, c2, h).permute(0, 2, 3, 1)      # (B, C2, H, W)
        return (gr + gc).flatten(2) / 2

    def forward(self, x):
        b, _, h, w = x.shape
        v = self.value(x).flatten(2)                       # B, C2, HW
        out = x.flatten(2)
        for _ in range(self.recurrent):
            q = self.q(x).flatten(2)
            k = self.key(x).flatten(2)
            criss = self._criss_attend(q, k, v, h, w)
            criss4 = criss.view(b, -1, h, w)
            gate = self.project(criss4).flatten(2)
            out = out + gate * (criss - out)
        return out.view(b, -1, h, w)


class ChannelAttention(nn.Module):
    """CAM: GAP -> FC -> ReLU -> FC -> sigmoid (used inside MS-GDFF fusion)."""

    def __init__(self, c, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, max(8, c // reduction)), nn.ReLU(inplace=True),
            nn.Linear(max(8, c // reduction), c), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x * self.fc(s).view(b, c, 1, 1)
