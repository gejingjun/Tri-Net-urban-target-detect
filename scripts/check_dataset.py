#!/usr/bin/env python3
"""Dataset sanity check (paper Sec. 4 statistics):
triplet alignment, invalid-depth fraction (~28.3% expected), pseudo-depth
fraction (~15.5% expected), per-batch counts vs Table 4, small-target ratios.

Usage: python scripts/check_dataset.py --root data/urbantraffic3m
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check(root: Path, split="train"):
    img_dir = root / "images" / split
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    miss = defaultdict(int)
    inv_frac, pseudo_n, small_n, total_n, invalid_box_img = [], 0, 0, 0, 0
    batch_count = defaultdict(int)
    for st in stems:
        ip = root / "images" / split / f"{st}.png"
        rp = root / "ir" / split / f"{st}.png"
        dp = root / "depth" / split / f"{st}.png"
        lp = root / "labels" / split / f"{st}.txt"
        for tag, p in (("rgb", ip), ("ir", rp), ("depth", dp), ("label", lp)):
            if not p.exists():
                miss[tag] += 1
        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[..., 0]
        inv_frac.append(float((d == 0).mean()))
        if d.max() <= 255 and d.dtype == np.uint8:
            pseudo_n += 1
        lb = (np.loadtxt(lp, dtype=np.float32, ndmin=2)
              if lp.exists() and lp.stat().st_size else np.zeros((0, 5)))
        for row in lb:
            w, h = row[3] * d.shape[1], row[4] * d.shape[0]
            total_n += 1
            if w * h < 32 ** 2:
                small_n += 1
        batch_count[st.split("_")[0]] += 1
    print(f"== {root} / {split} ==")
    print(f"images: {len(stems)}")
    print(f"missing triplet files: {dict(miss) or 'none'}")
    if inv_frac:
        print(f"mean invalid-depth pixel fraction: {np.mean(inv_frac):.3f} "
              f"(paper: 0.283)")
    print(f"pseudo-depth (uint8) images: {pseudo_n}/{len(stems)} = "
          f"{pseudo_n/max(len(stems),1):.3f} (paper: 0.155)")
    print(f"small targets (area<32^2): {small_n}/{total_n}")
    print("per-batch counts:", dict(batch_count))
    mp = root / "meta" / f"{split}.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        print(f"meta entries: {len(meta)} (batches+conditions tagged)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    check(Path(a.root), a.split)
