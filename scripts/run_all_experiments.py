#!/usr/bin/env python3
"""Reproduce ALL experiments and tables of the TFG-Net paper.

    Paper: TFG-Net (FGF-YOLO), Pattern Recognition, 2026
    Code:  https://github.com/gejingjun/Tri-Net-urban-target-detect

Each experiment = one YAML config (experiments/configs/...) x 3 seeds
{2026, 3407, 7102}; results are aggregated to mean±std and dumped to
experiments/logs/<name>_summary.json (mirrors the paper tables).

Usage
-----
# Everything (300 epochs x 3 seeds — needs data/urbantraffic3m, RTX 4090):
python scripts/run_all_experiments.py --set paper

# Fast end-to-end check of the whole pipeline on synthetic data:
python scripts/make_fake_data.py --out data/fake --n 24
python scripts/run_all_experiments.py --set quick   # ~minutes on 1 GPU

Individual table:  --set ablation / modality / sota / vade / encoding /
robustness / cross_dataset
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# ----------------------------------------------------------------------
# Table 6 — Modality necessity (7 rows)
MODALITY = ["rgb_only", "ir_only", "depth_only", "rgb_ir", "rgb_depth",
            "ir_depth", "ours"]
# Table 7 — SOTA comparison (TFG-Net row lives in tfgnet_urbantraffic3m.yaml)
SOTA = ["yolov8s_rgb", "yolov8s_ir", "early_fusion_concat",
        "degf_yolo", "cft_rgb_ir", "yolofusion_rgb_ir",
        "ours_wo_vade_gating"]
# Table 9 — Component ablation (5 rows)
ABLATION = ["00_baseline_rgbir", "01_sdfe", "02_sdfe_vade",
            "03_sdfe_vade_rugate", "04_full"]
# Table 10 — VADE phase decomposition (4 rows)
VADE = ["00_no_vade", "01_phase_a", "02_phase_b", "03_full"]
# Table 8 — Depth encoding comparison (5 rows)
ENCODING = ["00_raw", "01_nmf_10m_30m", "02_nmf_5m_11m", "03_inv_valid",
            "04_vade_a"]
# Table 4.5.2 — Systematic degradation (eval-only, needs the full model ckpt)
DEGRADATIONS = ["none", "depth_mask:0.3", "depth_mask:0.7", "depth_remove",
                "ir_remove", "rgb_remove", "ir_mask:0.5"]
CROSS = [("rgbdt500", "cross_dataset/rgbdt500.yaml"),
         ("dronevehicle_0shot", "cross_dataset/dronevehicle_zero_shot.yaml")]

TABLES = {"modality": (MODALITY, "modality"), "sota": (SOTA, "sota"),
          "ablation": (ABLATION, "ablation"), "vade": (VADE, "vade_phase"),
          "encoding": (ENCODING, "depth_encoding")}


def sh(cmd, env=None):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, env=env)


def run_cfg(cfg_path, seed, data_root, epochs, device):
    tmp = Path("experiments/configs/_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    raw = Path(cfg_path).read_text(encoding="utf-8")
    patched = raw.replace("data/urbantraffic3m", data_root)
    if epochs:
        patched += f"\ntrain:\n  epochs: {epochs}\n"
    p = tmp / Path(cfg_path).name
    p.write_text(patched, encoding="utf-8")
    sh([PY, "scripts/train.py", "--config", str(p), "--seed", seed])
    return p


def train_table(names, subdir, data_root, epochs, device, seeds):
    logs = Path("experiments/logs")
    logs.mkdir(parents=True, exist_ok=True)
    summary = {}
    for nm in names:
        cfg = ROOT / "experiments/configs" / subdir / f"{nm}.yaml"
        if not cfg.exists():
            print("SKIP missing", cfg); continue
        vals = []
        for sd in seeds:
            p = run_cfg(cfg, sd, data_root, epochs, device)
            from tfgnet.config import load_config as _lc
            run_name = _lc(p).name
            ck = ROOT / "runs" / f"{run_name}_seed{sd}" / "best.pt"
            import torch
            try:
                v = torch.load(ck, map_location="cpu", weights_only=False).get("res", {}).get("mAP50")
            except Exception:
                v = None
            if v is not None:
                vals.append(v)
        if len(vals) >= 2:
            summary[nm] = dict(mean=statistics.mean(vals),
                               std=statistics.stdev(vals), n=len(vals))
    (logs / "train_tables.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    print(json.dumps(summary, indent=2))


def robustness_table(ckpt, data_root, device):
    logs = Path("experiments/logs"); logs.mkdir(parents=True, exist_ok=True)
    rows = {}
    for deg in DEGRADATIONS:
        out = subprocess.run(
            [PY, "scripts/eval.py", "--config",
             "experiments/configs/tfgnet_urbantraffic3m.yaml",
             "--ckpt", ckpt, "--split", "test", "--degradation", deg,
             "--out-json", "experiments/logs/_deg_tmp.json"],
            capture_output=True, text=True)
        if Path("experiments/logs/_deg_tmp.json").exists():
            rows[deg] = json.loads(Path("experiments/logs/_deg_tmp.json")
                                   .read_text(encoding="utf-8"))["mAP50"]
    (logs / "degradation.json").write_text(json.dumps(rows, indent=2),
                                           encoding="utf-8")
    print("Table 4.5.2 degradation:", rows)


def condition_batch_tables(ckpt, cfg):
    out = subprocess.run([PY, "scripts/eval.py", "--config", cfg,
                          "--ckpt", ckpt, "--split", "test",
                          "--group-by", "condition",
                          "--out-json", "experiments/logs/_cond_tmp.json"],
                         capture_output=True, text=True)
    out = subprocess.run([PY, "scripts/eval.py", "--config", cfg,
                          "--ckpt", ckpt, "--split", "test",
                          "--group-by", "batch",
                          "--out-json", "experiments/logs/_batch_tmp.json"],
                         capture_output=True, text=True)
    for f in ("_cond_tmp.json", "_batch_tmp.json"):
        p = Path("experiments/logs") / f
        if p.exists():
            print(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", default="paper",
                    choices=["paper", "quick", "modality", "sota", "ablation",
                             "vade", "encoding", "robustness", "cross_dataset"])
    ap.add_argument("--data-root", default="data/urbantraffic3m")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs (quick mode: 2)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seeds = [2026, 3407, 7102]
    quick = args.set == "quick"
    if quick:
        args.data_root = "data/fake"
        args.epochs = 2
        seeds = [2026]

    if args.set in ("paper", "quick") or args.set in TABLES:
        which = (["modality", "sota", "ablation", "vade", "encoding"]
                 if args.set in ("paper", "quick") else [args.set])
        for w in which:
            names, subdir = TABLES[w]
            train_table(names, subdir, args.data_root, args.epochs,
                        args.device, seeds)

    if args.set in ("paper", "robustness") and not quick:
        ckpt = "runs/tfgnet_full_seed2026/best.pt"
        robustness_table(ckpt, args.data_root, args.device)
        condition_batch_tables(ckpt, "experiments/configs/tfgnet_urbantraffic3m.yaml")

    if args.set in ("paper", "cross_dataset") and not quick:
        for nm, cfg in CROSS:
            run_cfg(ROOT / "experiments/configs" / cfg, 2026,
                    args.data_root, None, args.device)

    print("All requested experiment sets finished. Summaries in experiments/logs/")


if __name__ == "__main__":
    main()
