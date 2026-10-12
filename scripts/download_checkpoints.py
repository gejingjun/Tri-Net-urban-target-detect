#!/usr/bin/env python3
"""Download the RGB-branch COCO pre-training weights (yolov8s.pt).

Paper Sec. 4.1.2: 'The RGB branch is initialized from COCO pre-training.'
If this file is missing, tfgnet warns and trains from scratch.

Usage: python scripts/download_checkpoints.py
"""
import sys
import urllib.request
from pathlib import Path

URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
]
OUT = Path(__file__).resolve().parents[1] / "checkpoints/yolov8s.pt"


def main():
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 1e6:
        print("already present:", OUT)
        return
    for u in URLS:
        try:
            print("downloading", u)
            urllib.request.urlretrieve(u, OUT)
            print("saved ->", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")
            return
        except Exception as e:
            print("  failed:", e)
    sys.exit("Could not fetch yolov8s.pt. Download it manually (see "
             "checkpoints/README.md) or run without COCO init.")


if __name__ == "__main__":
    main()
