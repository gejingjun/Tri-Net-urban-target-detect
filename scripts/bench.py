#!/usr/bin/env python3
"""Efficiency analysis (paper Sec. 4.5.4 table):
params, FLOPs, latency, FPS at 640x640 batch 1 on the target GPU, including
VADE Phase-A preprocessing and NMS.

Usage: python scripts/bench.py [--config experiments/configs/tfgnet_urbantraffic3m.yaml]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.config import load_config
from tfgnet.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/tfgnet_urbantraffic3m.yaml")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.model.rgb_pretrained = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.rand(1, 3, 640, 640, device=device)
    e = torch.rand(1, cfg.model.c_d, 640, 640, device=device)
    # FLOPs
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x, x, e), verbose=False)
        gflops = macs * 2 / 1e9
    except Exception:
        gflops = float("nan")
    # warmup + timing (preprocessing + forward + NMS)
    with torch.no_grad():
        for _ in range(5):
            det, _ = model(x, x, e)
            model.head.decode(det, conf_thres=0.001)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            det, _ = model(x, x, e)
            model.head.decode(det, conf_thres=0.001)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.iters * 1000
    print(f"{cfg.name}: params {params:.1f} M | FLOPs {gflops:.1f} G | "
          f"latency {ms:.1f} ms | FPS {1000/ms:.0f}  (batch 1, 640x640, "
          f"device={device.type})")
    # component breakdown (paper: depth branch 0.2M, fusion 1.6M, VADE-B 0.05M, RU-Gate 0.15M)
    def cnt(mod):
        return sum(p.numel() for p in mod.parameters()) / 1e6
    from tfgnet.model import TFGNet
    if isinstance(model, TFGNet):
        print(f"  rgb_bb {cnt(model.rgb_bb):.1f} | ir_bb {cnt(model.ir_bb):.1f} | "
              f"dep_bb {cnt(model.dep_bb) if model.dep_bb else 0:.1f} | "
              f"sdfe {sum(cnt(m) for m in model.sdfe):.2f} | "
              f"vade_b {cnt(model.vade_gate) if model.vade_gate else 0:.2f} | "
              f"msgdff {sum(cnt(m) for m in model.msgdff):.2f} | "
              f"rugate {sum(cnt(m) for m in model.rugate) if model.rugate else 0:.2f}")


if __name__ == "__main__":
    main()
