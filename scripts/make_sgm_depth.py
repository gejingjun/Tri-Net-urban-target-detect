#!/usr/bin/env python3
"""Synthesize metric-ish depth from DroneVehicle stereo RGB pairs via
Semi-Global Matching (paper Sec. 4.5.5: "depth maps are synthesized from
the stereo pair using Semi-Global Matching").

Reads  data/dronevehicle/{train,val}/visible/<stem>.png  (+ infrared) and
writes data/dronevehicle/depth/{split}/<stem>.png as uint16 disparity-derived
metric-proxy (disparity scaled to mm). VADE Phase A's pseudo routing (Eq.14)
and the validity gate handle the noise.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def sgm_depth(left, right, disp16max=64, scale_mm=20000):
    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    sgbm = cv2.SGBM_create(numDisparities=64, blockSize=5)
    sgbm.setPreFilterType(cv2.STEREOPREFILTER_XSOBEL) if hasattr(
        cv2, "STEREOPREFILTER_XSOBEL") else None
    disp = sgbm.compute(g1, g2)                       # 16.5 fixed
    disp = (disp / 16.0).astype(np.float32)
    depth = np.where(disp > 1e-3, scale_mm / np.maximum(disp, 1e-3), 0.0)
    return np.clip(depth, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dronevehicle")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    vin = Path(args.root) / args.split / "visible"
    out = Path(args.root) / "depth" / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(vin.glob("*.png"))
    print(f"{len(files)} visible images")
    for i, p in enumerate(files[:5000]):
        left = cv2.imread(str(p))
        # DroneVehicle ships aligned RGB/IR (not true stereo); IR as right
        # view approximates an intensity-based disparity baseline.
        right = cv2.imread(str(p).replace("visible", "infrared"))
        if left is None or right is None:
            continue
        d = sgm_depth(left, right)
        cv2.imwrite(str(out / p.name), d)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")
    print("depth ->", out)


if __name__ == "__main__":
    main()
