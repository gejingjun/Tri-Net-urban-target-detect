#!/usr/bin/env python3
"""Train TFG-Net / baselines.

Examples
--------
# Full paper protocol (300 epochs, batch 16, 3 seeds handled by run script):
python scripts/train.py --config experiments/configs/tfgnet_urbantraffic3m.yaml

# Quick sanity check on the synthetic sample set (CPU-friendly):
python scripts/make_fake_data.py --out data/fake --n 24
python scripts/train.py --config experiments/configs/smoke_fake.yaml

Resume:  python scripts/train.py --config ... --resume runs/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.config import load_config          # noqa: E402
from tfgnet.train import Trainer               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML experiment config")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override cfg.train.seeds[0]")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seeds = [args.seed]
    name = args.run_name or f"{cfg.name}_seed{cfg.train.seeds[0]}"
    tr = Trainer(cfg, run_name=name, resume=args.resume)
    best = tr.fit()
    print(f"[done] {cfg.name}: best mAP@50 = {best:.2f}")


if __name__ == "__main__":
    main()
