"""Losses (paper Sec. 3.6, Eq. 32).

    L = L_det + lambda_valid * L_valid + lambda_scene * L_scene

* L_det : standard detection loss (CIoU + BCE + DFL), YOLOv8-style
          TaskAlignedAssigner.
* L_valid = BCE(g_pixel, D_valid) — supervises the VADE Phase-B pixel gate
          against the binary validity indicator (Eq. 9), always correct by
          definition.
* L_scene : cross-entropy on scene labels when available, else 0.

Defaults: lambda_valid = 0.1, lambda_scene = 0.1 (Sec. 3.6).
Deliberately NO depth-reconstruction / cross-modal contrastive loss —
Sec. 3.6 explains why (invalid on the 28.3% structurally-absent pixels).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(b1, b2, CIoU=True, eps=1e-7):
    """Row-wise CIoU between two (N,4) xyxy sets."""
    x1, y1, x2, y2 = b1.unbind(-1)
    X1, Y1, X2, Y2 = b2.unbind(-1)
    inter = (x2.minimum(X2) - x1.maximum(X1)).clamp(min=0) * \
            (y2.minimum(Y2) - y1.maximum(Y1)).clamp(min=0)
    a = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    A = (X2 - X1).clamp(min=0) * (Y2 - Y1).clamp(min=0)
    u = a + A - inter + eps
    iou = inter / u
    if not CIoU:
        return iou
    cw = x2.maximum(X2) - x1.minimum(X1)
    ch = y2.maximum(Y2) - y1.minimum(Y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((x2 + x1 - X2 - X1) ** 2 + (y2 + y1 - Y2 - Y1) ** 2) / 4
    b1w, b1h = (x2 - x1).clamp(min=eps), (y2 - y1).clamp(min=eps)
    b2w, b2h = (X2 - X1).clamp(min=eps), (Y2 - Y1).clamp(min=eps)
    v = (4 / torch.pi ** 2) * (torch.atan(b2w / b2h) - torch.atan(b1w / b1h)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - rho2 / c2 - alpha * v


class TaskAlignedAssigner(nn.Module):
    """TAL assigner (YOLOv8): aligned score = cls^alpha * IoU^beta, top-k
    anchors per GT, anchor-center-inside-GT constraint, best-GT on conflict."""

    def __init__(self, topk=13, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk, self.alpha, self.beta = topk, alpha, beta

    @torch.no_grad()
    def __call__(self, cls_logits, pred_xyxy, gt_cls, gt_xyxy, gt_mask):
        """cls_logits (B,A,nc); pred_xyxy (B,A,4); gt_cls (B,G);
        gt_xyxy (B,G,4); gt_mask (B,G) bool.
        Returns fg_mask (B,A), matched_gt_idx (B,A), matched_cls (B,A)."""
        b, a, nc = cls_logits.shape
        g = gt_cls.shape[1]
        iou = _batch_iou(pred_xyxy, gt_xyxy)                        # (B,A,G)
        cls_prob = cls_logits.sigmoid()                             # (B,A,nc)
        onehot = F.one_hot(gt_cls.clamp(min=0), nc).float()          # (B,G,nc)
        align = (cls_prob @ onehot.transpose(1, 2)) ** self.alpha \
            * iou ** self.beta                                       # (B,A,G)
        n_gt = gt_mask.sum(1)                                        # (B,)
        tk = n_gt.clamp(max=self.topk)
        in_gt = _centers_inside(pred_xyxy, gt_xyxy) & gt_mask[:, None, :]  # (B,A,G)
        scores = align * in_gt.float()
        topv, topi = scores.topk(self.topk, dim=1)                   # (B,k,G)
        rank_ok = torch.arange(self.topk, device=scores.device)[None, :, None] < tk.view(b, 1, 1)
        sel = torch.zeros_like(scores)
        sel.scatter_(1, topi, ((topv > 0) & gt_mask.unsqueeze(1) & rank_ok).float())
        # anchors claimed by >1 gt: resolve to the max-align gt (vectorised)
        align_masked = align * sel
        matched = sel.amax(2) > 0                                     # (B,A)
        matched_gt = align_masked.argmax(2)                           # (B,A)
        # GTS without any anchor: fallback to their best in-box anchor
        empty_gt = gt_mask & (sel.sum(1) == 0)                        # (B,G)
        if empty_gt.any():
            fb_scores = (scores + align * 0.0) * empty_gt[:, None, :].float()
            fb_anchor = (align * in_gt.float() * empty_gt[:, None, :].float()).amax(2, keepdim=True)
            fa = (align * in_gt.float() * empty_gt[:, None, :].float()).argmax(1)  # (B,G)
            bi, gj = empty_gt.nonzero().unbind(1)
            has_in = (align * in_gt.float())[bi, :, gj].amax(1) > 0
            take = bi[has_in], gj[has_in], fa[bi[has_in], gj[has_in]]
            matched[take[0], take[2]] = True
            matched_gt[take[0], take[2]] = take[1]
        matched_cls = gt_cls.gather(1, matched_gt.clamp(min=0))         # (B,A)
        matched_cls = torch.where(matched, matched_cls.clamp(min=0),
                                  torch.zeros_like(matched_cls))
        return matched, matched_gt, matched_cls


def _batch_iou(a, b):
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    iw = (ax2[..., None].minimum(bx2[:, None, :]) - ax1[..., None].maximum(bx1[:, None, :])).clamp(min=0)
    ih = (ay2[..., None].minimum(by2[:, None, :]) - ay1[..., None].maximum(by1[:, None, :])).clamp(min=0)
    inter = iw * ih
    aa = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[..., None]
    ab = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[:, None, :]
    return inter / (aa + ab - inter + 1e-9)


def _centers_inside(boxes, gt):
    c = (boxes[..., :2] + boxes[..., 2:]) / 2
    return ((c[..., 0:1] >= gt[:, None, :, 0]) & (c[..., 0:1] <= gt[:, None, :, 2]) &
            (c[..., 1:2] >= gt[:, None, :, 1]) & (c[..., 1:2] <= gt[:, None, :, 3]))


class DetectionLoss(nn.Module):
    """CIoU + BCE + DFL over 3 levels (YOLOv8 protocol, weighted)."""

    def __init__(self, nc=12, reg_max=16, strides=(8, 16, 32), img_size=640,
                   lb=7.5, lc=0.5, ld=1.5):
        super().__init__()
        self.nc, self.reg_max, self.strides, self.img = nc, reg_max, list(strides), img_size
        self.assigner = TaskAlignedAssigner()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("arange", torch.arange(reg_max, dtype=torch.float32))
        self.lb, self.lc, self.ld = lb, lc, ld

    def anchors(self, feat_sizes, device):
        cents, strs = [], []
        for (h, w), s in zip(feat_sizes, self.strides):
            gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                    torch.arange(w, device=device, dtype=torch.float32),
                                    indexing="ij")
            cents.append(torch.stack((gx + 0.5, gy + 0.5), -1).reshape(-1, 2) * s)
            strs += [s] * (h * w)
        return torch.cat(cents), torch.tensor(strs, device=device)

    def decode(self, reg_raw, cents, strs):
        b = reg_raw.shape[0]
        dist = reg_raw.view(b, 4, self.reg_max, -1).softmax(2)                 # (B,4,R,A)
        ltrb = (dist * self.arange.view(1, 1, self.reg_max, 1)).sum(2)        # (B,4,A)
        st = strs[None, :, None]                                                # (1,A,1)
        cx = cents[None, :, 0:1]; cy = cents[None, :, 1:2]                     # (1,A,1)
        l, t_, r, b_ = (ltrb[:, i].unsqueeze(-1) for i in range(4))            # (B,A,1)
        x1 = cx - l * st; y1 = cy - t_ * st
        x2 = cx + r * st; y2 = cy + b_ * st
        return torch.cat([x1, y1, x2, y2], -1), dist                           # (B,A,4)

    def forward(self, outs, targets):
        device = outs[0][0].device
        b = outs[0][0].shape[0]
        feats = [(o[0].shape[2], o[0].shape[3]) for o in outs]
        cents, strs = self.anchors(feats, device)
        cls = torch.cat([o[0].flatten(2) for o in outs], 2)                  # (B,nc,A)
        reg = torch.cat([o[1].flatten(2) for o in outs], 2)                  # (B,4R,A)
        cls_t = cls.permute(0, 2, 1)                                           # (B,A,nc)
        pred_xyxy, dist = self.decode(reg, cents, strs)
        gt_cls = targets["cls"].to(device).long()
        gt_n = targets["n"].to(device)
        g = gt_cls.shape[1]
        boxes_n = targets["boxes"].to(device)                                   # (B,G,4) cxcywh norm
        ok = (torch.arange(g, device=device)[None] < gt_n[:, None])
        gb = boxes_n * self.img
        gt_xyxy = torch.cat([gb[..., :2] - gb[..., 2:] / 2,
                             gb[..., :2] + gb[..., 2:] / 2], -1)
        gt_xyxy = torch.where(ok[..., None], gt_xyxy, torch.zeros_like(gt_xyxy))
        gt_cls_s = torch.where(ok, gt_cls.clamp(min=0), torch.zeros_like(gt_cls))
        fg, gi, gc = self.assigner(cls_t, pred_xyxy, gt_cls_s, gt_xyxy, ok)
        nfg = fg.sum()
        out = {"loss_cls": cls_t.new_zeros(()), "loss_box": cls_t.new_zeros(()),
               "loss_dfl": cls_t.new_zeros(()), "n_fg": nfg}
        if nfg > 0:
            tgt = torch.zeros_like(cls_t)
            bi, ai = fg.nonzero().unbind(1)
            tgt[bi, ai, gc[fg]] = 1.0
            out["loss_cls"] = self.bce(cls_t, tgt).sum(-1).mean() * self.lc
            sel_box = torch.gather(gt_xyxy, 1, gi.unsqueeze(-1).expand(-1, -1, 4))[fg]
            pre_box = pred_xyxy[fg]
            iou = bbox_iou(pre_box, sel_box, CIoU=True)
            out["loss_box"] = (1 - iou).mean() * self.lb
            # DFL: soft one-hot CE on ltrb targets expressed in stride units
            pos_c = (pre_box[..., :2] + pre_box[..., 2:]) / 2
            strs_f = strs.unsqueeze(0).expand(b, -1)[fg]
            lt = (sel_box[..., :2] - pos_c) / strs_f[:, None]
            rb = (sel_box[..., 2:] - pos_c) / strs_f[:, None]
            t = torch.cat([lt, rb], 1).clamp(0, self.reg_max - 1 - 0.01)      # (N,4)
            tl = t.long(); tr = tl + 1
            dl = tr.float() - t; dr = t - tl.float()
            d4 = dist.permute(0, 3, 1, 2)                                    # (B,A,4,R)
            di = fg.nonzero()
            dv = d4[di[:, 0], di[:, 1]]                                      # (N,4,R)
            logv = dv.clamp(min=1e-7).log()
            l_tl = logv.gather(2, tl.unsqueeze(-1)).squeeze(-1)
            l_tr = logv.gather(2, tr.unsqueeze(-1)).squeeze(-1)
            out["loss_dfl"] = (-(dl * l_tl + dr * l_tr)).mean() * self.ld
        return out


class AuxiliaryLosses(nn.Module):
    """L_valid (Eq.9 vs g_pixel) + L_scene (Eq.29a) — Sec. 3.6."""

    def __init__(self, lambda_valid=0.1, lambda_scene=0.1):
        super().__init__()
        self.lv, self.ls = lambda_valid, lambda_scene

    def validity(self, g_pixel, d_raw_valid):
        """g_pixel (B,1,Hf,Wf); d_raw_valid (B,1,H,W) Eq.9 indicator, resized."""
        if g_pixel.shape[-2:] != d_raw_valid.shape[-2:]:
            d = F.interpolate(d_raw_valid, size=g_pixel.shape[-2:],
                              mode="nearest")
        else:
            d = d_raw_valid
        return F.binary_cross_entropy(g_pixel.clamp(1e-6, 1 - 1e-6), d) * self.lv

    def scene(self, scene_probs, scene_labels):
        if scene_labels is None or scene_labels.min() < 0:
            return scene_probs.sum() * 0.0
        logits = scene_probs.clamp(1e-6, 1 - 1e-6).log()
        return F.nll_loss(logits, scene_labels.to(logits.device)) * self.ls
