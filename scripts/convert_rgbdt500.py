#!/usr/bin/env python3
"""Convert RGBDT500 (official release) to the repo YOLO-style tri-modal layout.

RGBDT500 ships aligned RGB / Depth (uint16, mm) / Thermal triplets with per
sequence annotation JSON or txt files (release-dependent). This script
normalises paths and label format; run it once after extraction:

    python scripts/convert_rgbdt500.py --src /path/to/RGBDT500 \
        --dst data/rgbdt500 --split train
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def find_triplets(src: Path):
    """Locate <stem> keys: RGB image + depth + thermal in the release tree."""
    rgb_exts = {".png", ".jpg"}
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in rgb_exts and ("visible" in str(p.parent)
                                             or "rgb" in str(p.parent).lower()):
            yield p


def convert(src: Path, dst: Path, split: str):
    for sub in ("images", "ir", "depth", "labels"):
        (dst / sub / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for rp in find_triplets(src):
        seq = rp.parent
        stem = f"{seq.name}_{rp.stem}"
        dp = next((seq / f"{rp.stem}{e}" for e in (".png", ".tiff")
                   if (seq / "depth" / f"{rp.stem}{e}").exists()),
                  seq / "depth" / f"{rp.stem}.png")
        tp = seq / "thermal" / f"{rp.stem}.png"
        if not (tp.exists() or Path(dp).exists()):
            continue
        shutil.copy(rp, dst / "images" / split / f"{stem}.png")
        shutil.copy(tp, dst / "ir" / split / f"{stem}.png")
        depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if depth is not None:
            cv2.imwrite(str(dst / "depth" / split / f"{stem}.png"), depth)
        # labels: RGBDT500 uses [cls, x1, y1, x2, y2] in its own txt/json;
        # convert to YOLO cxcywh normalised
        for lab_name in (f"{rp.stem}.txt", f"{rp.stem}.json"):
            lab = seq / "labels" / lab_name
            if lab.exists():
                rows = []
                if lab.suffix == ".json":
                    ann = json.loads(lab.read_text(encoding="utf-8"))
                    objs = ann.get("objects", ann if isinstance(ann, list) else [])
                    for o in objs:
                        bb = o["bbox"]
                        rows.append([o["type"], (bb[0]+bb[2])/2, (bb[1]+bb[3])/2,
                                     bb[2]-bb[0], bb[3]-bb[1]])
                else:
                    for line in lab.read_text(encoding="utf-8").splitlines():
                        v = [float(x) for x in re.split(r"[\s,]+", line) if x]
                        if len(v) >= 5:
                            rows.append([v[0], (v[1]+v[3])/2, (v[2]+v[4])/2,
                                         v[3]-v[1], v[4]-v[2]])
                if rows:
                    arr = np.array(rows, np.float32)
                    arr[:, [2, 4]] /= 640.0
                    arr[:, [1, 3]] /= 640.0
                    np.savetxt(dst / "labels" / split / f"{stem}.txt", arr,
                               fmt="%.5f")
                break
        n += 1
    print(f"converted {n} triplets -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/rgbdt500")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.dst), a.split)
