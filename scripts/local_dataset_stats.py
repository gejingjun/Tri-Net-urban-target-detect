#!/usr/bin/env python3
"""Full-corpus statistics on the LOCAL UrbanTraffic-3M copy — produces the
dataset-fact numbers for the paper checklist (Sec. 4.1.1 / Table 4 / the
28.3% / 15.5% / per-class / small-target statistics).

Read-only over the dataset; writes experiments/logs/local_dataset_stats.json.

Usage: python scripts/local_dataset_stats.py [--root data/urbantraffic3m]
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
               "car", "ball", "light", "garbagecan", "uav", "tricycle"]
# paper Sec. 4.1.1 instance counts, for reference in the printed table
PAPER_INSTANCES = {"person": 5499, "animal": 3061, "car": 1571, "light": 1455,
                   "sign": 809, "bicycle": 666, "seat": 621,
                   "garbagecan": 282, "uav": 204, "boat": 135, "ball": 89,
                   "tricycle": 25}


def imread_u(p, flag=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, flag)


def batch_of(stem):
    for b in ("seq6_3_8", "seq_suppl", "shuming", "hehe"):
        if stem.startswith(b):
            return b
    return "numeric" if stem[0].isdigit() else stem.split("_")[0]


def analyze_split(root, split):
    img_dir = root / "images" / split
    dep_dir = root / "depth" / split
    lab_dir = root / "labels" / split
    stems = sorted({p.stem for p in img_dir.iterdir()})
    inv, pseudo, dtypes = [], 0, Counter()
    box_inv = 0            # boxes with all-invalid depth under them
    box_total = 0
    per_cls = Counter()
    small = 0
    inst_total = 0
    batch_cnt = Counter()
    for st in stems:
        batch_cnt[batch_of(st)] += 1
        dp = next((dep_dir / f"{st}{e}" for e in (".png", ".jpg", ".jpeg")
                   if (dep_dir / f"{st}{e}").exists()), None)
        ip = next((img_dir / f"{st}{e}" for e in (".png", ".jpg", ".jpeg")
                   if (img_dir / f"{st}{e}").exists()), None)
        if dp is None or ip is None:
            continue
        dep = imread_u(dp)
        rgb = imread_u(ip, cv2.IMREAD_COLOR)
        if dep is None or rgb is None:
            continue
        if dep.ndim == 3:
            dep = dep[..., 0]
        dtypes[str(dep.dtype)] += 1
        inv.append(float((dep == 0).mean()))
        if int(dep.max()) <= 255 and dep.dtype == np.uint8:
            pseudo += 1
        H, W = rgb.shape[:2]
        lp = lab_dir / f"{st}.txt"
        if lp.exists() and lp.stat().st_size:
            rows = np.loadtxt(lp, dtype=np.float32, ndmin=2)
            if rows.ndim == 1:
                rows = rows[None]
            inst_total += len(rows)
            per_cls.update(rows[:, 0].astype(int).tolist())
            for r in rows:
                box_total += 1
                x1 = int((r[1] - r[3] / 2) * W); y1 = int((r[2] - r[4] / 2) * H)
                x2 = int((r[1] + r[3] / 2) * W); y2 = int((r[2] + r[4] / 2) * H)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, W), min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                frac = (r[3] * W) * (r[4] * H)
                if frac < 32 ** 2:
                    small += 1
                roi = dep[y1:y2, x1:x2]
                if roi.size and (roi == 0).mean() >= 0.999:
                    box_inv += 1
    return dict(
        n_images=len(stems),
        invalid_depth_pixel_frac_mean=float(np.mean(inv)),
        pseudo_depth_images=pseudo,
        pseudo_depth_frac=pseudo / max(len(stems), 1),
        depth_dtypes=dict(dtypes),
        images_with_zero_depth_box_frac=box_inv / max(box_total, 1),
        box_zero_depth=box_inv, box_total=box_total,
        instances=inst_total, small_instances=small,
        small_frac=small / max(inst_total, 1),
        per_class={int(k): (CLASS_NAMES[int(k)] if int(k) < 12 else str(k), v)
                   for k, v in sorted(per_cls.items())},
        batches=dict(batch_cnt),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/urbantraffic3m")
    a = ap.parse_args()
    root = Path(a.root)
    out = {}
    for split in ("train", "test"):
        if not (root / "images" / split).exists():
            continue
        out[split] = analyze_split(root, split)
        s = out[split]
        print(f"\n== {split} ({s['n_images']} images) ==")
        print(f"  invalid-depth pixel frac : {s['invalid_depth_pixel_frac_mean']*100:.2f}%")
        print(f"  pseudo-depth images      : {s['pseudo_depth_images']} ({s['pseudo_depth_frac']*100:.2f}%)")
        print(f"  depth dtypes             : {s['depth_dtypes']}")
        print(f"  images with >=1 all-invalid box : {s['images_with_zero_depth_box_frac']*100:.2f}%"
              f"  (boxes {s['box_zero_depth']}/{s['box_total']})")
        print(f"  instances / small(<32^2) : {s['instances']} / {s['small_instances']}"
              f" ({s['small_frac']*100:.2f}%)")
        print(f"  batches                  : {s['batches']}")
        print("  per-class instances:")
        for k, (name, v) in sorted(s["per_class"].items(), key=lambda x: -x[1][1]):
            ref = PAPER_INSTANCES.get(name, "?")
            print(f"    {name:11s} {v:6d}   (paper: {ref})")
    Path("experiments/logs").mkdir(exist_ok=True)
    Path("experiments/logs/local_dataset_stats.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> experiments/logs/local_dataset_stats.json")


if __name__ == "__main__":
    main()
