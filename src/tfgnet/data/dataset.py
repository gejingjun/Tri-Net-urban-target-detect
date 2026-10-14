"""Tri-modal dataset (UrbanTraffic-3M layout, RGBDT500, DroneVehicle, fake).

Directory layout (YOLO-style labels, per-image depth PNG):

    root/
      images/{split}/*.png          RGB, uint8, 3ch
      ir/{split}/*.png              IR (effectively grayscale), uint8 1- or 3-ch
      depth/{split}/*.png           depth: uint16 mm (metric) OR uint8 pseudo
      labels/{split}/*.txt          "cls cx cy w h" normalised 0-1
      meta/{split}.json (optional)  {"<stem>": {"batch": "seq6_3_8",
                                               "condition": "fog"}}

VADE Phase A encoding (Eqs. 9-14) is deterministic dataset-side
pre-processing; the dataset returns the encoded 6-channel tensor E_dep plus
the raw validity indicator (target of L_valid, Eq. 9/32).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.vade import DepthEncoder


def _imread(path: Path, flag):
    # np.fromfile + imdecode: unicode-safe on Windows (cv2.imread on CJK paths)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(path)
    return img


IMG_EXTS = (".png", ".jpg", ".jpeg")


def rng_sample(v, k, rng):
    """Uniform sub-sample without replacement (numpy 1-D array)."""
    idx = rng.sample(range(len(v)), k)
    return v[idx]


class TriModalDataset(Dataset):
    def __init__(self, root: str | Path, split: str, img_size=640,
                 encoder: DepthEncoder | None = None,
                 augmentor=None, nc=12, stride_mult=32, max_images=None):
        self.root = Path(root)
        self.split = split
        self.img = img_size
        self.nc = nc
        self.aug = augmentor
        self.enc = encoder or DepthEncoder()
        self.stride = stride_mult
        self.samples = []
        seen = set()
        for ext in IMG_EXTS:
            for p in sorted((self.root / "images" / split).glob(f"*{ext}")):
                if p.stem in seen:
                    continue
                seen.add(p.stem)
                lp = self.root / "labels" / split / f"{p.stem}.txt"
                if lp.exists():
                    self.samples.append(p.stem)
        self.samples.sort()
        if max_images is not None:
            self.samples = self.samples[:max_images]
        meta_p = self.root / "meta" / f"{split}.json"
        self.meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    def __len__(self):
        return len(self.samples)

    def bind_mosaic_pool(self):
        """Give the augmentor access to other samples (mosaic neighbours).
        Uses a bound method (pickle-able across DataLoader workers), not a lambda."""
        if self.aug is not None and hasattr(self.aug, "set_pool"):
            self.aug.set_pool(self._load_raw, len(self))

    def _load_raw(self, i):
        rgb, ir, dep, lb, _ = self._load(i)
        return rgb, ir, dep, lb

    # ------------------------------------------------------------------
    def _p(self, mod: str, stem: str) -> Path:
        for ext in IMG_EXTS:
            p = self.root / mod / self.split / f"{stem}{ext}"
            if p.exists():
                return p
        return self.root / mod / self.split / f"{stem}.png"

    def _load(self, i):
        stem = self.samples[i]
        rgb = _imread(self._p("images", stem), cv2.IMREAD_COLOR)
        irp = self._p("ir", stem)
        ir = _imread(irp, cv2.IMREAD_GRAYSCALE) if irp.exists() else rgb[..., 0]
        dep = _imread(self._p("depth", stem), cv2.IMREAD_UNCHANGED)
        if dep.ndim == 3:
            dep = dep[..., 0]
        lb_path = self.root / "labels" / self.split / f"{stem}.txt"
        lb = (np.loadtxt(lb_path, dtype=np.float32, ndmin=2)
              if lb_path.exists() and lb_path.stat().st_size > 0
              else np.zeros((0, 5), np.float32))
        return rgb, ir, dep, lb, stem

    def _resize(self, rgb, ir, dep, boxes):
        h, w = rgb.shape[:2]
        sx, sy = self.img / w, self.img / h
        rgb = cv2.resize(rgb, (self.img, self.img), interpolation=cv2.INTER_LINEAR)
        ir = cv2.resize(ir, (self.img, self.img), interpolation=cv2.INTER_LINEAR)
        # depth: nearest (keep metric scale exact)
        dep = cv2.resize(dep.astype(np.float32), (self.img, self.img),
                         interpolation=cv2.INTER_NEAREST)
        if boxes.size:
            b = boxes.copy()
            b[:, [1, 3]] *= sx
            b[:, [2, 4]] *= sy
            boxes = b
        return rgb, ir, dep, boxes

    # ------------------------------------------------------------------
    def __getitem__(self, i):
        rgb, ir, dep, lb, stem = self._load(i)
        if self.aug is not None:
            rgb, ir, dep, lb = self.aug(rgb, ir, dep, lb)
        rgb, ir, dep, lb = self._resize(rgb, ir, dep, lb)
        # pad to stride multiple
        pw, ph = (-self.img) % self.stride, (-self.img) % self.stride
        if pw or ph:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
            ir = np.pad(ir, ((0, ph), (0, pw)))
            dep = np.pad(dep.astype(np.float32), ((0, ph), (0, pw)))
            lb = lb.copy() if lb.size else lb
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0)
        ir_arr = ir if ir.ndim == 3 else np.stack([ir] * 3, -1)
        ir_t = torch.from_numpy(np.ascontiguousarray(ir_arr[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0)
        e_dep = self.enc.encode(torch.from_numpy(dep.astype(np.float32)))
        valid = (dep > 0).astype(np.float32)
        valid_t = torch.from_numpy(valid)[None]
        n = len(lb)
        MAXB = 300
        lb = lb[:MAXB]
        n = len(lb)
        boxes = torch.zeros(MAXB, 4)  # cxcywh normalised (of padded size)
        cls = torch.full((MAXB,), -1, dtype=torch.long)
        H, W = rgb.shape[:2]
        if n:
            cx = lb[:, 1] / W; cy = lb[:, 2] / H; bw = lb[:, 3] / W; bh = lb[:, 4] / H
            boxes[:n] = torch.tensor(np.stack([cx, cy, bw, bh], 1), dtype=torch.float32)
            cls[:n] = torch.tensor(lb[:, 0].astype(np.int64))
        meta = self.meta.get(stem, {})
        return dict(rgb=rgb_t, ir=ir_t, e_dep=e_dep, valid=valid_t,
                    boxes=boxes, cls=cls, n=n,
                    stem=stem, batch=meta.get("batch", "unknown"),
                    condition=meta.get("condition", "unknown"),
                    # depth stats for router/VADE debug
                    depth_invalid_frac=float(1.0 - valid.mean()))

    # ------------------------------------------------------------------
    def fit_encoder(self, max_images=200, max_pixels=2_000_000):
        """Collect valid pixels from the split and fit VADE-A quantiles (Eq. 11).

        Pixels are randomly sub-sampled to <= max_pixels (torch.quantile has a
        ~16 M element limit and would be far too slow/memory-heavy otherwise).
        """
        import random
        rng = random.Random(2026)
        vals = []
        idx = list(range(len(self)))
        rng.shuffle(idx)
        cap = max(1, max_pixels // max(1, len(idx)))
        for i in idx[: min(len(idx), max_images)]:
            _, _, dep, _, _ = self._load(i)
            v = dep[dep > 0].astype(np.float32).ravel()
            if v.size > cap:
                v = rng_sample(v, cap, rng)
            vals.append(v if v.size else np.zeros(1, np.float32))
        allv = np.concatenate(vals)[:max_pixels]
        self.enc.fit(torch.from_numpy(allv))
        return self.enc


def collate_tri(batch):
    out = {}
    for k in ("rgb", "ir", "e_dep", "valid", "boxes", "cls"):
        out[k] = torch.stack([b[k] for b in batch])
    out["n"] = torch.tensor([int(b["n"]) for b in batch], dtype=torch.long)
    for k in ("stem", "batch", "condition", "depth_invalid_frac"):
        out[k] = [b[k] for b in batch]
    return out
