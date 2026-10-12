"""Tiny helpers shared across the repo: seeding, logging, IoU, paths."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs. Paper reports mean±std over seeds
    {2026, 3407, 7102} (Training details, Sec. 3.6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "tfgnet", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def xyxy_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes [N,4] and [M,4] -> [N,M]."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = torch.clamp(ax2[:, None] - bx1[None, :], min=0)
    ih = torch.clamp(ay2[:, None] - by1[None, :], min=0)
    inter = iw * ih
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def area_of_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Pixel area of [N,4] xyxy boxes."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def clamp_label(label: np.ndarray, nc: int) -> np.ndarray:
    return np.clip(label, 0, nc - 1)
