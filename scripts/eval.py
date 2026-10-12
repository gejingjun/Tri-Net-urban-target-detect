#!/usr/bin/env python3
"""Evaluate a trained checkpoint.

Examples
--------
python scripts/eval.py --config experiments/configs/tfgnet_urbantraffic3m.yaml \
    --ckpt runs/tfgnet_seed2026/best.pt

# Robustness table (Sec. 4.5.2): systematic modality degradation
python scripts/eval.py --config ... --ckpt ... --degradation depth_remove
python scripts/eval.py --config ... --ckpt ... --degradation depth_mask:0.3
python scripts/eval.py --config ... --ckpt ... --degradation ir_mask:0.5

# Per-condition / per-batch breakdowns (Sec. 4.5.3)
python scripts/eval.py --config ... --ckpt ... --group-by condition
python scripts/eval.py --config ... --ckpt ... --group-by batch

# Late-fusion ensemble baseline (Table Main Comparison row)
python scripts/eval.py --config experiments/configs/late_fusion_rgb.yaml ... \
    --late-fusion ckts rgb.pt ir.pt depth.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.config import load_config                    # noqa: E402
from tfgnet.data.dataset import TriModalDataset          # noqa: E402
from tfgnet.evaluation import dump_coco_json, evaluate_with_areas  # noqa: E402
from tfgnet.model import SingleBranchNet, TFGNet         # noqa: E402
from tfgnet.modules.head import xyxy_nms                 # noqa: E402
from tfgnet.modules.vade import DepthEncoder             # noqa: E402


DEG_MODES = ("depth_mask", "depth_remove", "ir_remove", "rgb_remove", "ir_mask")


def parse_degradation(s):
    """'depth_mask:0.3' -> {'depth_mask': 0.3}; 'depth_remove' -> {'depth_remove': 1}"""
    if not s or s == "none":
        return None
    if ":" in s:
        k, v = s.split(":")
        return {k: float(v)}
    return {s: 1}


@torch.no_grad()
def run_eval(cfg, model, ds, device, enc=None):
    preds, gts = [], []
    tags = []
    model.eval()
    for i in range(len(ds)):
        b = ds[i]
        rgb = b["rgb"][None].to(device); ir = b["ir"][None].to(device)
        e = b["e_dep"][None].to(device)
        if isinstance(model, TFGNet):
            det, _ = model(rgb, ir, e)
            per = model.head.decode(det, conf_thres=1e-3)
            bxs, scs, cls = per[0]
            k = xyxy_nms(bxs, scs, cls, iou_thres=0.65)
        else:
            det, _ = model(rgb)
            per = model.head.decode(det, conf_thres=1e-3)
            bxs, scs, cls = per[0]
            k = xyxy_nms(bxs, scs, cls, iou_thres=0.65)
        preds.append(dict(image_id=b["stem"], boxes=bxs[k].cpu().numpy(),
                          scores=scs[k].cpu().numpy(), cls=cls[k].cpu().numpy()))
        n = int(b["n"])
        boxes = b["boxes"][:n].numpy() * cfg.data.img_size
        xyxy = torch.from_numpy(boxes)
        xyxy = torch.cat([xyxy[:, :2] - xyxy[:, 2:] / 2,
                          xyxy[:, :2] + xyxy[:, 2:] / 2], 1)
        gts.append(dict(image_id=b["stem"], boxes=xyxy.numpy(),
                        cls=b["cls"][:n].numpy()))
        tags.append((b["batch"], b["condition"]))
    res = evaluate_with_areas(preds, gts)
    return res, preds, gts, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=None, help="default: cfg.data.val_split")
    ap.add_argument("--degradation", default=None,
                    help="none|depth_mask:0.3|depth_mask:0.7|depth_remove|ir_remove|rgb_remove|ir_mask:0.5")
    ap.add_argument("--group-by", choices=["condition", "batch"], default=None)
    ap.add_argument("--dump-coco-pred", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.degradation = parse_degradation(args.degradation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt
    if not Path(ckpt_path).exists():
        alt = Path(ckpt_path).with_name("last.pt")
        if alt.exists():
            print(f"[warn] {ckpt_path} not found -> using {alt}")
            ckpt_path = str(alt)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode = getattr(cfg.model, "depth_encoding_mode", "vade_a")
    enc = DepthEncoder(cfg.model.c_d, mode=mode,
                       fixed_edges=cfg.model.fixed_edges_mm if mode == "near_mid_far" else None)
    if "encoder" in ck:
        enc.load_state_dict(ck["encoder"])
    model = TFGNet(cfg).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)

    ds = TriModalDataset(cfg.data.root, args.split or cfg.data.val_split,
                         cfg.data.img_size, enc, None, nc=cfg.data.nc)
    if args.max_images:
        ds.samples = ds.samples[: args.max_images]
    res, preds, gts, tags = run_eval(cfg, model, ds, device)
    report = {"config": cfg.name, "degradation": cfg.degradation,
              **{k: v for k, v in res.items() if k != "per_class"},
              "per_class": {str(k): round(v, 2) for k, v in res["per_class"].items()}}
    if args.group_by:
        idx = 0 if args.group_by == "batch" else 1
        import numpy as np
        groups = {}
        for p, g, t in zip(preds, gts, tags):
            groups.setdefault(t[idx], ([], []))
            groups[t[idx]][0].append(p)
            groups[t[idx]][1].append(g)
        report["grouped"] = {k: round(evaluate_with_areas(v[0], v[1])["mAP50"], 2)
                             for k, v in sorted(groups.items())}
    if args.dump_coco_pred:
        dump_coco_json(preds, args.dump_coco_pred, cfg.data.names)
        report["coco_pred"] = args.dump_coco_pred
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
