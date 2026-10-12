#!/usr/bin/env python3
"""Visualizations (paper Sec. 4.6, Figs. 8-9):

  * Grad-CAM on the fused feature (which regions the network attends to)
  * per-modality quality/reliability maps G_vis / G_ir / G_dep and the
    VADE pixel-validity gate g_pixel
  * RU-Gate mode masks (which of the three modes fires where)

Usage:
  python scripts/visualize.py --config experiments/configs/tfgnet_urbantraffic3m.yaml \
      --ckpt runs/tfgnet_full_seed2026/best.pt \
      --images data/urbantraffic3m/images/val --out results/figures
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.config import load_config
from tfgnet.data.dataset import TriModalDataset
from tfgnet.model import TFGNet
from tfgnet.modules.vade import DepthEncoder


def gradcam(model, rgb, ir, e):
    """Class-agnostic Grad-CAM w.r.t. the fused P3 feature."""
    feats = {}
    def hook(_, i, o):
        t = o[0] if isinstance(o, tuple) else (o["f_fuse"] if isinstance(o, dict) else o)
        feats["p3"] = t                      # keep in-graph for autograd.grad
    target = model.rugate[0] if model.rugate is not None else model.msgdff[0]
    h = target.register_forward_hook(hook)
    try:
        det, aux = model(rgb, ir, e)
        score = det[0][0].sigmoid().max(1)[0].sum()
        grad = torch.autograd.grad(score, feats["p3"])[0]
    finally:
        h.remove()
    w = grad.mean((2, 3), keepdim=True)
    cam = (w * feats["p3"]).sum(1).relu()[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.detach().cpu().numpy()


def overlay(img, heat):
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    hm = plt.get_cmap("jet")(heat)[..., :3] * 255
    return cv2.addWeighted(img, 0.6, hm.astype(np.uint8), 0.4, 0)


@torch.no_grad()
def reliability_maps(model, rgb, ir, e):
    """Return (g_pixel, G_vis, G_ir, G_dep, m_deg, m_con) at P3."""
    outs = {}
    def hook(mod, inp, o):
        outs["msgdff"] = o
    def hook2(mod, inp, o):
        outs["rugate"] = o
    hs = [model.msgdff[0].register_forward_hook(hook),
          model.rugate[0].register_forward_hook(hook2)]
    try:
        model.forward(rgb, ir, e)
    finally:
        for h in hs:
            h.remove()
    g = model.vade_gate.last_g_pixel[0, 0].cpu().numpy()
    m = outs["msgdff"]
    gq = {k: m[f"g_{k}"][0, 0].cpu().numpy() for k in ("vis", "ir", "dep")}
    info = outs["rugate"][1]
    return g, gq, info["m_deg"][0, 0].cpu().numpy(), info["m_con"][0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", default=None, help="default: dataset val split")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--degradation", default=None, help="e.g. ir_remove to show gate adaptation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.degradation:
        k, _, v = args.degradation.partition(":")
        cfg.degradation = {k: float(v) if v else 1.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TFGNet(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    enc = DepthEncoder(cfg.model.c_d)
    if "encoder" in ck:
        enc.load_state_dict(ck["encoder"])
    ds = TriModalDataset(cfg.data.root, cfg.data.val_split, cfg.data.img_size, enc,
                         None, nc=cfg.data.nc)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stems = sorted(Path(args.images).glob("*.png"))[: args.n] if args.images else None
    for i in range(min(args.n, len(ds))):
        b = ds[i]
        rgb = b["rgb"][None].to(device); ir = b["ir"][None].to(device)
        e = b["e_dep"][None].to(device)
        rgb_h = rgb[0].cpu(); ir_h = ir[0].cpu(); e_h = e[0].cpu()
        base = (rgb_h.permute(1, 2, 0).numpy()[..., ::-1] * 255).astype(np.uint8)
        cam = gradcam(model, rgb, ir, e)
        g_pix, gq, m_deg, m_con = reliability_maps(model, rgb, ir, e)
        fig, axes = plt.subplots(1, 7, figsize=(26, 4))
        titles = ["RGB", "IR", "depth(valid)", "Grad-CAM", "g_pixel",
                  "G_vis/G_ir/G_dep", "RU-Gate deg|con"]
        axes[0].imshow(base); axes[0].set_title(titles[0])
        axes[1].imshow(ir_h[0].numpy(), cmap="gray")
        axes[1].set_title(titles[1])
        axes[2].imshow(e_h[0].numpy(), cmap="gray"); axes[2].set_title(titles[2])
        axes[3].imshow(overlay(base, cam)); axes[3].set_title(titles[3])
        axes[4].imshow(g_pix, cmap="magma"); axes[4].set_title(titles[4])
        blend = np.stack([gq["vis"], gq["ir"], gq["dep"]], -1)
        axes[5].imshow(blend); axes[5].set_title(titles[5])
        axes[6].imshow(np.stack([m_deg, m_con, np.zeros_like(m_deg)], -1))
        axes[6].set_title(titles[6])
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"viz_{b['stem']}.png", dpi=120)
        plt.close(fig)
    print("figures ->", out.resolve())


if __name__ == "__main__":
    main()
