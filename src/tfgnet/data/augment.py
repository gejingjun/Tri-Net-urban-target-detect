"""Tri-modal synchronized augmentation (paper Sec. 3.6 Training details).

    Mosaic (applied IDENTICALLY to all three modalities), MixUp,
    random horizontal flip, HSV jitter (RGB; brightness-only for IR,
    geometry-only for depth).

All geometry ops keep RGB / IR / depth pixel-aligned, which the model
assumes (Limitation (i), Sec. 5).
"""
from __future__ import annotations

import cv2
import numpy as np


class TriModalAugmentor:
    def __init__(self, mosaic=True, mixup=0.15, hflip=0.5,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, prob=0.8, img_size=640,
                 rng=None):
        self.mosaic, self.mixup, self.hflip = mosaic, mixup, hflip
        self.hsv = (hsv_h, hsv_s, hsv_v)
        self.prob = prob
        self.img = img_size
        self.rng = rng or np.random.default_rng(0)

    # ------------------------------------------------------------------
    def __call__(self, rgb, ir, dep, lb):
        if self.mosaic and self.rng.random() < self.prob:
            rgb, ir, dep, lb = self._mosaic(rgb, ir, dep, lb)
        if self.rng.random() < self.hflip:
            rgb, ir, dep, lb = self._hflip(rgb, ir, dep, lb)
        rgb, ir, dep, lb = self._resize_keep_aspect(rgb, ir, dep, lb)
        rgb, ir = self._photometric(rgb, ir)
        return rgb, ir, dep, lb

    # ------------------------------------------------------------------
    def _mosaic(self, rgb, ir, dep, lb):
        """4-quadrant mosaic built from this sample plus 3 random neighbours
        fetched through the injected loader (applied IDENTICALLY to all
        three modalities). Falls back to letterbox when no loader exists."""
        loader = getattr(self, "_loader", None)
        pool_size = getattr(self, "_pool_size", 0)
        if loader is None or pool_size == 0:
            return self._resize_keep_aspect(rgb, ir, dep, lb)
        s = self.img // 2
        canvas_rgb = np.full((self.img, self.img, 3), 114, np.uint8)
        canvas_ir = np.full((self.img, self.img), 114, np.uint8)
        canvas_dep = np.zeros((self.img, self.img), np.float32)
        canvas_lb = []
        quads = [(0, 0), (s, 0), (0, s), (s, s)]
        # first quad is the current image
        imgs = [(rgb, ir, dep, lb)]
        for _ in range(3):
            imgs.append(loader(int(self.rng.integers(pool_size))))
        for (x0, y0), (r, i_, d, lbx) in zip(quads, imgs):
            h, w = r.shape[:2]
            k = s / max(w, h)
            rr = cv2.resize(r, (int(w * k), int(h * k)))
            ii = cv2.resize(i_, (int(w * k), int(h * k)))
            dd = cv2.resize(d.astype(np.float32), (int(w * k), int(h * k)),
                            interpolation=cv2.INTER_NEAREST)
            canvas_rgb[y0:y0 + rr.shape[0], x0:x0 + rr.shape[1]] = rr
            canvas_ir[y0:y0 + ii.shape[0], x0:x0 + ii.shape[1]] = ii
            canvas_dep[y0:y0 + dd.shape[0], x0:x0 + dd.shape[1]] = dd
            if lbx.size:
                b = lbx.copy()
                b[:, 1] = b[:, 1] * w * k + x0
                b[:, 2] = b[:, 2] * h * k + y0
                b[:, 3] = b[:, 3] * w * k
                b[:, 4] = b[:, 4] * h * k
                canvas_lb.append(b)
        lb_all = np.concatenate(canvas_lb, 0) if canvas_lb else np.zeros((0, 5), np.float32)
        # normalise to the mosaic canvas size
        if lb_all.size:
            lb_all[:, [1, 3]] /= self.img
            lb_all[:, [2, 4]] /= self.img
        return canvas_rgb, canvas_ir, canvas_dep, lb_all

    def set_pool(self, loader, pool_size):
        """loader(idx) -> (rgb, ir, dep, lb); used to fetch mosaic neighbours."""
        self._loader = loader
        self._pool_size = int(pool_size)

    # ------------------------------------------------------------------
    def _resize_keep_aspect(self, rgb, ir, dep, lb):
        h, w = rgb.shape[:2]
        k = self.img / max(w, h)
        nw, nh = int(w * k), int(h * k)
        rgb = cv2.resize(rgb, (nw, nh))
        ir = cv2.resize(ir, (nw, nh))
        dep = cv2.resize(dep.astype(np.float32), (nw, nh),
                         interpolation=cv2.INTER_NEAREST)
        if lb.size:
            b = lb.copy()
            pw, ph = self.img - nw, self.img - nh
            b[:, [1, 3]] *= nw / w
            b[:, [2, 4]] *= nh / h
            b[:, 1] += pw / 2 / self.img
            b[:, 2] += ph / 2 / self.img
            lb = b
        pad_b = ((0, self.img - nh), (0, self.img - nw))
        rgb = np.pad(rgb, (*pad_b, (0, 0)), constant_values=114)
        ir = np.pad(ir, pad_b, constant_values=114)
        dep = np.pad(dep, pad_b)
        return rgb, ir, dep, lb

    def _hflip(self, rgb, ir, dep, lb):
        rgb, ir, dep = rgb[:, ::-1], ir[:, ::-1], dep[:, ::-1]
        if lb.size:
            b = lb.copy()
            b[:, 1] = 1.0 - b[:, 1]
            lb = b
        return np.ascontiguousarray(rgb), np.ascontiguousarray(ir), \
            np.ascontiguousarray(dep), lb

    def _photometric(self, rgb, ir):
        gh, gs, gv = self.hsv
        # RGB: full HSV jitter
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.rng.uniform(-gh, gh) * 180) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + self.rng.uniform(-gs, gs)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + self.rng.uniform(-gv, gv)), 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # IR (grayscale): brightness only
        ir = np.clip(ir.astype(np.float32) * (1 + self.rng.uniform(-gv, gv)), 0, 255).astype(np.uint8)
        return rgb, ir


def mixup_pair(a, b, lam_rng):
    """MixUp between two augmented samples (all modalities, same lambda)."""
    lam = float(lam_rng.beta(0.5, 0.5))
    out = {}
    for k in ("rgb", "ir", "e_dep"):
        out[k] = lam * a[k] + (1 - lam) * b[k]
    out["valid"] = torch.maximum(a["valid"], b["valid"]) if hasattr(a["valid"], "maximum") \
        else np.maximum(a["valid"], b["valid"])
    n = int(a["n"]) + int(b["n"])
    out["boxes"] = torch.cat([a["boxes"][:a["n"]], b["boxes"][:b["n"]]])[:50]
    out["cls"] = torch.cat([a["cls"][:a["n"]], b["cls"][:b["n"]]])[:50]
    out["n"] = min(n, 50)
    return out
