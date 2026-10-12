"""FPN+PAN neck and YOLOv8 decoupled anchor-free head (paper Sec. 4.1.2).

Neck consumes the three RU-Gate outputs F_out^{P3,P4,P5} and produces the
same three levels for the head, exactly as in Algorithm 1 (last two lines).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C3k2, Conv


class Neck(nn.Module):
    """Top-down FPN + bottom-up PAN over P3/P4/P5."""

    def __init__(self, ch=(128, 256, 512), depth_mult=0.33):
        super().__init__()
        d = lambda n: max(1, int(round(n * depth_mult)))
        c3, c4, c5 = ch
        # top-down
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat4 = Conv(c5, c4, 1)
        self.p4_out = C3k2(c4 + c4, c4, d(3), True)
        self.lat3 = Conv(c4, c3, 1)
        self.p3_out = C3k2(c3 + c3, c3, d(3), True)
        # bottom-up
        self.down3 = Conv(c3, c3, 3, 2)
        self.p4_pan = C3k2(c3 + c4, c4, d(3), True)
        self.down4 = Conv(c4, c4, 3, 2)
        self.p5_pan = C3k2(c4 + c5, c5, d(3), False)
        self.out_ch = (c3, c4, c5)

    def forward(self, p3, p4, p5):
        n4 = self.p4_out(torch.cat([self.up(self.lat4(p5)), p4], 1))
        n3 = self.p3_out(torch.cat([self.up(self.lat3(n4)), p3], 1))
        n4b = self.p4_pan(torch.cat([self.down3(n3), n4], 1))
        n5b = self.p5_pan(torch.cat([self.down4(n4b), p5], 1))
        return n3, n4b, n5b


class DFL(nn.Module):
    """Distribution focal loss -> ltrb (as in YOLOv8)."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data.copy_(x.view(1, c1, 1, 1).requires_grad_(False))
        self.c1 = c1

    def forward(self, x):
        # x: (B, 4*c1, h, w) channel-major [l bins, t bins, r bins, b bins]
        b, _, h, w = x.shape
        d = x.view(b, 4, self.c1, h * w).softmax(2)          # normalise over bins
        idx = self.conv.weight.view(1, 1, self.c1, 1)        # bins [0..c1-1]
        return (d * idx).sum(2).view(b, 4, h, w)


class DetectHead(nn.Module):
    """YOLOv8-style decoupled anchor-free head on 3 levels.

    Outputs, per level: cls logits (B,nc,A), reg ltrb raw (B,4,A).
    ``decode`` turns raw predictions into xyxy boxes + scores for NMS.
    """

    def __init__(self, ch=(128, 256, 512), nc=12, reg_max=16, stride=(8, 16, 32)):
        super().__init__()
        self.nc, self.reg_max, self.stride = nc, reg_max, stride
        self.cv_cls = nn.ModuleList(nn.Sequential(
            Conv(c, max(c // 2, 32), 3), nn.Conv2d(max(c // 2, 32), nc, 1)) for c in ch)
        self.cv_reg = nn.ModuleList(nn.Sequential(
            Conv(c, max(c // 2, 32), 3), nn.Conv2d(max(c // 2, 32), 4 * reg_max, 1)) for c in ch)
        self.dfl = DFL(reg_max)

    def forward(self, feats):
        outs = []
        for f, cc, rc in zip(feats, self.cv_cls, self.cv_reg):
            outs.append((cc(f), rc(f)))
        return outs

    @torch.no_grad()
    def decode(self, outs, img_size=640, conf_thres=0.25):
        """-> per-batch list of (boxes (N,4) xyxy px, scores (N,), cls (N,))."""
        b = outs[0][0].shape[0]
        per = [[] for _ in range(b)]
        for li, (cls_raw, reg_raw) in enumerate(outs):
            _, _, h, w = cls_raw.shape
            s = self.stride[li]
            ltrb = self.dfl(reg_raw)                                  # (B,4,h,w)
            yy, xx = torch.meshgrid(
                torch.arange(h, device=cls_raw.device, dtype=torch.float32),
                torch.arange(w, device=cls_raw.device, dtype=torch.float32), indexing="ij")
            cx = (xx + 0.5) * s
            cy = (yy + 0.5) * s
            x1 = cx - ltrb[:, 0] * s
            y1 = cy - ltrb[:, 1] * s
            x2 = cx + ltrb[:, 2] * s
            y2 = cy + ltrb[:, 3] * s
            xyxy = torch.stack([x1, y1, x2, y2], 1).flatten(2).transpose(1, 2)  # (B,A,4)
            conf = cls_raw.flatten(2).transpose(1, 2).sigmoid()       # (B,A,nc)
            score, cid = conf.max(-1)
            for bi in range(b):
                k = score[bi] > conf_thres
                if k.any():
                    per[bi].append((xyxy[bi][k], score[bi][k], cid[bi][k]))
        out = []
        for bi in range(b):
            if per[bi]:
                boxes, scores, cls = zip(*per[bi])
                out.append((torch.cat(boxes), torch.cat(scores), torch.cat(cls)))
            else:
                dev = outs[0][0].device
                out.append((torch.zeros(0, 4, device=dev),
                            torch.zeros(0, device=dev),
                            torch.zeros(0, dtype=torch.long, device=dev)))
        return out


def xyxy_nms(boxes, scores, classes, iou_thres=0.65, max_det=300):
    """Class-wise greedy NMS. Returns indices into the original arrays."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    keep_all = []
    for ci in classes.unique():
        m = (classes == ci).nonzero().flatten()
        b, s = boxes[m], scores[m]
        order = s.argsort(descending=True)
        sel = _nms_single(b[order], s[order], iou_thres)
        keep_all.append(m[torch.tensor(sel, dtype=torch.long, device=m.device)])
    keep = torch.cat(keep_all)
    final = scores[keep].argsort(descending=True)[:max_det]
    return keep[final]


def _nms_single(b, s, iou_thres):
    """Greedy NMS over score-sorted boxes; returns positions in b."""
    if b.shape[0] == 0:
        return []
    x1, y1, x2, y2 = b.T
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    n = b.shape[0]
    suppressed = torch.zeros(n, dtype=torch.bool, device=b.device)
    keep = []
    for i in range(n):
        if bool(suppressed[i]):
            continue
        keep.append(i)
        iw = (x2.clamp(max=x2[i]) - x1.clamp(min=x1[i])).clamp(min=0)
        ih = (y2.clamp(max=y2[i]) - y1.clamp(min=y1[i])).clamp(min=0)
        inter = iw * ih
        iou = inter / (area[i] + area - inter + 1e-9)
        suppressed |= iou > iou_thres
    return keep
