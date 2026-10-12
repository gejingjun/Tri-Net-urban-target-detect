#!/usr/bin/env python3
"""Materialise every experiment config under experiments/configs/.

Re-run after changing base.yaml defaults. Each generated YAML carries a
'base: base.yaml' pointer and only the deltas for that experiment row.
"""
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "experiments" / "configs"
OUT.mkdir(parents=True, exist_ok=True)

BASE = {
    "name": "tfgnet",
    "model": {},  # defaults live in tfgnet.config
    "data": {"root": "data/urbantraffic3m", "train_split": "train",
             "val_split": "val"},
    "train": {"epochs": 300, "batch": 16, "seeds": [2026, 3407, 7102],
              "save_dir": "runs"},
}


def w(relpath: str, doc: dict):
    doc = {"base": "base.yaml" if relpath != "base.yaml" else None, **doc}
    doc = {k: v for k, v in doc.items() if v is not None}
    p = OUT / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")


# ---------------------------------------------------------------- base
w("base.yaml", BASE)
w("tfgnet_urbantraffic3m.yaml", {"name": "tfgnet_full",
                                 "comment": "Paper headline model (Table 7, ours)"})

# ---------------------------------------------------------------- smoke
w("smoke_fake.yaml", {
    "name": "smoke_fake",
    "data": {"root": "data/fake"},
    "train": {"epochs": 2, "batch": 2, "warmup_epochs": 0, "workers": 0,
              "amp": False, "seeds": [2026]},
    "model": {"rgb_pretrained": None},
})

# ---------------------------------------------------------------- Table 6
# Modality necessity (7 rows). Missing modalities -> zeroed inputs.
MOD = {
    "rgb_only":   {"modalities": ["rgb"]},
    "ir_only":    {"modalities": ["ir"]},
    "depth_only": {"modalities": ["depth"]},
    "rgb_ir":     {"modalities": ["rgb", "ir"]},
    "rgb_depth":  {"modalities": ["rgb", "depth"]},
    "ir_depth":   {"modalities": ["ir", "depth"]},
    "ours":       {"modalities": ["rgb", "ir", "depth"]},
}
for k, v in MOD.items():
    zero = {"rgb": v["modalities"], }
    w(f"modality/{k}.yaml", {"name": f"modality_{k}", "model": v})

# ---------------------------------------------------------------- Table 7
# Main comparison baselines
w("sota/yolov8s_rgb.yaml", {"name": "yolov8s_rgb", "model": {
    "naive_early_fusion": False, "modalities": ["rgb"],
    "use_depth": False, "use_sdfe": False, "use_vade_a": False,
    "use_vade_b": False, "use_rugate": False}})
w("sota/yolov8s_ir.yaml", {"name": "yolov8s_ir", "model": {
    "modalities": ["ir"], "use_depth": False, "use_sdfe": False,
    "use_vade_a": False, "use_vade_b": False, "use_rugate": False}})
w("sota/early_fusion_concat.yaml", {"name": "early_fusion",
                                    "model": {"naive_early_fusion": True}})
w("sota/late_fusion.yaml", {"name": "late_fusion",
                            "model": {"late_fusion": True},
                            "eval": {"protocol": "ensemble3"}})
w("sota/degf_yolo.yaml", {"name": "degf_yolo", "model": {
    "degf_style": True, "use_depth": False, "use_vade_a": False,
    "use_vade_b": False, "use_rugate": False, "modalities": ["rgb", "ir"]}})
w("sota/cft_rgb_ir.yaml", {"name": "cft_rgb_ir", "model": {
    "use_depth": False, "use_sdfe": False, "use_vade_b": False,
    "use_rugate": False, "modalities": ["rgb", "ir"]},
    "comment": "CFT-style cross-attention baseline approximated by fused "
               "bimodal MSGDFF without SDFE/VADE (see docs/REPRODUCE.md)"})
w("sota/yolofusion_rgb_ir.yaml", {"name": "yolofusion_rgb_ir", "model": {
    "use_depth": False, "use_sdfe": False, "use_vade_b": False,
    "use_rugate": False, "modalities": ["rgb", "ir"]}})
w("sota/ours_wo_vade_gating.yaml", {"name": "ours_wo_vade",
                                    "model": {"use_vade_b": False}})

# ---------------------------------------------------------------- Table 9
# Component ablation (additive over YOLOv8s RGB+IR baseline)
w("ablation/00_baseline_rgbir.yaml", {"name": "abl00", "model": {
    "use_sdfe": False, "use_vade_a": False, "use_vade_b": False,
    "use_rugate": False, "use_depth": False, "modalities": ["rgb", "ir"]}})
w("ablation/01_sdfe.yaml", {"name": "abl01", "model": {
    "use_sdfe": True, "use_vade_a": False, "use_vade_b": False,
    "use_rugate": False, "use_depth": False, "modalities": ["rgb", "ir"]}})
w("ablation/02_sdfe_vade.yaml", {"name": "abl02", "model": {
    "use_sdfe": True, "use_vade_a": True, "use_vade_b": True,
    "use_rugate": False, "use_depth": True}})
w("ablation/03_sdfe_vade_rugate.yaml", {"name": "abl03", "model": {
    "use_sdfe": True, "use_vade_a": True, "use_vade_b": True,
    "use_rugate": True, "use_depth": False}})
w("ablation/04_full.yaml", {"name": "abl04_full", "model": {}})

# ---------------------------------------------------------------- Table 10
# VADE phase decomposition
w("vade_phase/00_no_vade.yaml", {"name": "vade00", "model": {
    "use_vade_a": False, "use_vade_b": False}})
w("vade_phase/01_phase_a.yaml", {"name": "vade01", "model": {
    "use_vade_a": True, "use_vade_b": False}})
w("vade_phase/02_phase_b.yaml", {"name": "vade02", "model": {
    "use_vade_a": False, "use_vade_b": True,
    "depth_encoding_mode": "raw"}})
w("vade_phase/03_full.yaml", {"name": "vade03", "model": {}})

# ---------------------------------------------------------------- Table 8
# Depth encoding comparison (all with VADE Phase B gating on)
w("depth_encoding/00_raw.yaml", {"name": "enc_raw", "model": {
    "depth_encoding_mode": "raw", "use_vade_a": False}})
w("depth_encoding/01_nmf_10m_30m.yaml", {"name": "enc_10_30", "model": {
    "depth_encoding_mode": "near_mid_far", "use_vade_a": False,
    "fixed_edges_mm": [10000.0, 30000.0]}})
w("depth_encoding/02_nmf_5m_11m.yaml", {"name": "enc_5_11", "model": {
    "depth_encoding_mode": "near_mid_far", "use_vade_a": False,
    "fixed_edges_mm": [5000.0, 11000.0]}})
w("depth_encoding/03_inv_valid.yaml", {"name": "enc_inv_valid", "model": {
    "depth_encoding_mode": "inv_valid", "use_vade_a": False}})
w("depth_encoding/04_vade_a.yaml", {"name": "enc_vade_a", "model": {}})

# ---------------------------------------------------------------- Tables 11/12
# degradation & condition/batch: eval-side switches (scripts/eval.py flags)
w("robustness/degradation_eval.yaml", {"name": "degradation_eval",
    "comment": "Run scripts/eval.py --degradation {depth_mask:0.3,depth_mask:0.7,"
               "depth_remove,ir_remove,rgb_remove,ir_mask:0.5} plus the "
               "no-VADE-PhaseB variant via experiments/configs/"
               "sota/ours_wo_vade_gating.yaml"})
w("robustness/condition_batch.yaml", {"name": "cond_batch",
    "comment": "scripts/eval.py --group-by condition|batch on the test split"})

# ---------------------------------------------------------------- Tables 13/14
w("cross_dataset/rgbdt500.yaml", {"name": "rgbdt500", "model": {},
    "data": {"root": "data/rgbdt500", "nc": 66, "train_split": "train",
             "val_split": "test"}})
w("cross_dataset/dronevehicle_zero_shot.yaml", {"name": "dronevehicle_0shot",
    "data": {"root": "data/dronevehicle", "nc": 5, "train_split": "train",
             "val_split": "val"},
    "zero_shot_from": "runs/tfgnet_full_seed2026/best.pt"})

print("configs written to", OUT)
for p in sorted(OUT.rglob("*.yaml")):
    print("  ", p.relative_to(OUT.parent.parent))
