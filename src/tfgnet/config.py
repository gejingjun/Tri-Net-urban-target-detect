"""YAML-backed configuration (dataclass -> dict merge).

Every experiment in the paper is a config in experiments/configs/*.yaml;
`scripts/run_all_experiments.py` iterates them. Flags reproduce Table
"Modality necessity" / "Component ablation" / "VADE phase decomposition" /
"Depth encoding comparison" / degradation & transfer settings.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelConfig:
    # module toggles (ablation knobs) -------------------------------------
    use_sdfe: bool = True
    use_vade_a: bool = True          # Phase A quantile encoding
    use_vade_b: bool = True          # Phase B validity gating
    use_rugate: bool = True
    use_depth: bool = True
    modalities: List[str] = field(default_factory=lambda: ["rgb", "ir", "depth"])
    naive_early_fusion: bool = False   # concat raw depth (baseline row)
    late_fusion: bool = False          # per-modality detectors + NMS ensemble
    degf_style: bool = False           # DEGF-YOLO-style (uniform DFE + bimodal GDFF), no depth
    # widths / depths -------------------------------------------------------
    base: str = "YOLOv8s"
    width_mult: float = 0.5
    depth_mult: float = 0.33
    depth_branch_chans: List[int] = field(default_factory=lambda: [16, 32, 64])
    c_d: int = 6                    # Cd encoded depth channels (paper §3.3.1)
    quantiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    d_min_mm: float = 300.0
    d_max_mm: float = 19999.0
    region_k: int = 16
    p_drop: float = 0.15
    theta_low: float = 0.1
    theta_high: float = 0.25
    theta_diff: float = 0.15
    n_scenes: int = 5
    rgb_pretrained: str | None = "checkpoints/yolov8s.pt"
    # Table 8 (Depth Encoding Comparison): vade_a|raw|near_mid_far|inv_valid
    depth_encoding_mode: str = "vade_a"
    fixed_edges_mm: List[float] = field(default_factory=lambda: [10000.0, 30000.0])

    def __post_init__(self):
        # keep c_d consistent with the chosen depth-encoding mode (Table 8)
        if self.depth_encoding_mode != "vade_a":
            self.c_d = {"raw": 1, "near_mid_far": 3, "inv_valid": 2}[
                self.depth_encoding_mode]


@dataclass
class DataConfig:
    root: str = "data/urbantraffic3m"
    train_split: str = "train"
    val_split: str = "val"
    img_size: int = 640
    nc: int = 12
    # Competition-official id order (matches 初赛数据集 labels/*.txt);
    # paper Sec. 4.1.1 lists the same 12 classes in a different display order.
    names: List[str] = field(default_factory=lambda: [
        "person", "boat", "animal", "seat", "sign", "bicycle",
        "car", "ball", "light", "garbagecan", "uav", "tricycle"])
    depth_dtype: str = "auto"        # auto: metric uint16 | pseudo uint8 (Eq.14)
    max_images: int | None = None    # optional subset cap (smoke on real data)
    val_max_images: int | None = None  # separate cap for the eval split
    augment: dict = field(default_factory=lambda: {
        "mosaic": True, "mixup": 0.15, "hflip": 0.5, "hsv_h": 0.015,
        "hsv_s": 0.7, "hsv_v": 0.4})
    stats_json: str | None = None    # dataset stats (quantile taus, etc.)


@dataclass
class TrainConfig:
    epochs: int = 300
    batch: int = 16
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: int = 10
    scheduler: str = "cosine"
    lambda_valid: float = 0.1
    lambda_scene: float = 0.1
    seeds: List[int] = field(default_factory=lambda: [2026, 3407, 7102])
    device: str = "auto"
    workers: int = 4
    accum: int = 1          # grad accumulation; batch*accum = effective batch
    eval_every: int = 10
    amp: bool = True
    grad_clip: float = 10.0
    save_dir: str = "runs"


@dataclass
class Config:
    name: str = "tfgnet"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # experiment hooks ------------------------------------------------------
    degradation: dict | None = None    # e.g. {"depth_mask": 0.3}
    condition_filter: str | None = None  # e.g. "fog"
    batch_filter: str | None = None      # e.g. "seq6_3_8"
    zero_shot_from: str | None = None    # cross-dataset transfer eval

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = copy.deepcopy(d)
        for k in list(d.keys()):
            if k not in ("name", "model", "data", "train", "degradation",
                         "condition_filter", "batch_filter", "zero_shot_from",
                         "comment", "eval"):
                d.pop(k)
        return Config(
            name=d.pop("name", "tfgnet"),
            model=ModelConfig(**d.pop("model", {})),
            data=DataConfig(**d.pop("data", {})),
            train=TrainConfig(**d.pop("train", {})),
            **{k: v for k, v in d.items() if k in ("degradation", "condition_filter",
                                                   "batch_filter", "zero_shot_from")})


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    if base:
        bp = p.parent / base
        probe = p.parent
        while not bp.exists() and probe.parent != probe:
            probe = probe.parent
            bp = probe / base
        braw = yaml.safe_load(bp.read_text(encoding="utf-8")) or {} if bp.exists() else {}
        braw.pop("base", None)
        merged = _deep_merge(braw, raw)
    else:
        merged = raw
    return Config.from_dict(merged)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
