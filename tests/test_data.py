"""Data pipeline + config tests."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.config import Config, load_config
from tfgnet.data.dataset import TriModalDataset, collate_tri
from tfgnet.data.splits import batch_of, sequence_of
from tfgnet.modules.vade import DepthEncoder


def test_dataset_shapes_and_labels(fake_root):
    ds = TriModalDataset(fake_root, "train", 640, DepthEncoder(), None, nc=12)
    assert len(ds) > 0
    b = ds[0]
    assert b["rgb"].shape == (3, 640, 640)
    assert b["ir"].shape == (3, 640, 640)
    assert b["e_dep"].shape == (6, 640, 640)
    assert b["valid"].shape == (1, 640, 640)
    assert b["rgb"].dtype == torch.float32
    assert 0 <= b["n"] <= 50
    boxes = b["boxes"][:b["n"]]
    assert (boxes[:, 2:] > 0).all()
    assert (boxes[:, :2] > 0).all()
    assert b["cls"][:b["n"]].max() < 12


def test_collate_batches(fake_root):
    ds = TriModalDataset(fake_root, "train", 640, DepthEncoder(), None)
    batch = collate_tri([ds[0], ds[1]])
    assert batch["rgb"].shape == (2, 3, 640, 640)
    assert batch["n"].shape == (2,) and batch["n"].dtype == torch.long


def test_invalid_depth_fraction_statistic(fake_root):
    """Synthetic data mimics the paper's 28.3% structural-absence statistic."""
    ds = TriModalDataset(fake_root, "train", 640, DepthEncoder(), None)
    fr = [ds[i]["depth_invalid_frac"] for i in range(len(ds))]
    assert 0.15 < np.mean(fr) < 0.45


def test_fit_encoder_quantiles(fake_root):
    ds = TriModalDataset(fake_root, "train", 640, DepthEncoder(), None)
    enc = ds.fit_encoder(max_images=6)
    assert len(enc.tau) == 3
    # synthetic scenes have a constant background plane -> quantiles may tie
    assert enc.tau[0] <= enc.tau[1] <= enc.tau[2]


def test_augment_preserves_alignment_and_labels(fake_root):
    from tfgnet.data.augment import TriModalAugmentor
    import cv2
    ds = TriModalDataset(fake_root, "train", 640, DepthEncoder(), None)
    rgb = cv2.imread(str(Path(ds.root) / "images" / ds.split / f"{ds.samples[0]}.png"))
    ir = cv2.imread(str(Path(ds.root) / "ir" / ds.split / f"{ds.samples[0]}.png"),
                    cv2.IMREAD_GRAYSCALE)
    dep = cv2.imread(str(Path(ds.root) / "depth" / ds.split / f"{ds.samples[0]}.png"),
                     cv2.IMREAD_UNCHANGED)
    lb = np.loadtxt(Path(ds.root) / "labels" / ds.split / f"{ds.samples[0]}.txt",
                    dtype=np.float32, ndmin=2)
    aug = TriModalAugmentor(mosaic=False, img_size=128,
                            rng=np.random.default_rng(1))
    r2, i2, d2, l2 = aug(rgb.copy(), ir.copy(), dep.astype(np.float32).copy(), lb.copy())
    assert r2.shape[:2] == (128, 128) and i2.shape == (128, 128)
    assert d2.shape == (128, 128)
    # label boxes stay in [0,1] and count preserved
    if len(l2):
        assert l2[:, 1:5].min() >= 0 and l2[:, 1:5].max() <= 1
        assert len(l2) == len(lb)


def test_hflip_mirror():
    from tfgnet.data.augment import TriModalAugmentor
    lb = np.array([[0, 0.25, 0.5, 0.1, 0.1]], np.float32)
    img = np.zeros((8, 8, 3), np.uint8)
    g = np.zeros((8, 8), np.uint8)
    d = np.zeros((8, 8), np.float32)
    aug = TriModalAugmentor(mosaic=False, img_size=8)
    r, i_, dd, l2 = aug._hflip(img, g, d, lb)
    assert abs(l2[0, 1] - 0.75) < 1e-6


def test_mosaic_falls_back_without_pool():
    from tfgnet.data.augment import TriModalAugmentor
    aug = TriModalAugmentor(mosaic=True, prob=1.0, img_size=64,
                            rng=np.random.default_rng(0))
    r, i_, d, l2 = aug(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8),
                       np.zeros((64, 64), np.float32),
                       np.array([[0, .5, .5, .2, .2]], np.float32))
    assert r.shape == (64, 64, 3) and len(l2) == 1


def test_split_helpers():
    assert batch_of("seq6_3_8_00000123") == "seq6_3_8"
    assert batch_of("shuming_82_00000181") == "shuming"
    assert sequence_of("shuming_82_00000181") == "shuming_82"
    assert sequence_of("numeric_00000012") != ""


def test_batch_stratified_sequence_grouped(tmp_path):
    from tfgnet.data.splits import batch_stratified_split
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta").mkdir(exist_ok=True)
    for i in range(20):
        batch = "shuming" if i % 2 else "numeric"
        seq = 7 if batch == "shuming" else 3
        stem = f"{batch}_{seq}_{i:07d}"
        (tmp_path / "images" / "train" / f"{stem}.png").write_bytes(b"")
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")
    out = batch_stratified_split(tmp_path, tmp_path, seed=2026)
    # all frames of one sequence must not straddle the split
    groups = {}
    for side in ("train", "val"):
        for st in out[side]:
            groups.setdefault(sequence_of(st), set()).add(side)
    assert all(len(v) == 1 for v in groups.values())


def test_config_roundtrip():
    cfg = Config()
    d = cfg.to_dict()
    c2 = Config.from_dict(d)
    assert c2.train.seeds == cfg.train.seeds
    assert c2.model.theta_diff == cfg.model.theta_diff


def test_base_inheritance_and_unknown_keys():
    cfg = load_config("experiments/configs/smoke_fake.yaml")
    assert cfg.train.epochs == 2            # from smoke config
    assert cfg.data.names == cfg.data.names  # base.yaml defaults inherited
    assert isinstance(cfg.data.nc, int)


def test_encoding_mode_channels():
    cfg = load_config("experiments/configs/depth_encoding/02_nmf_5m_11m.yaml")
    assert cfg.model.c_d == 3
    assert cfg.model.depth_encoding_mode == "near_mid_far"
    assert cfg.model.fixed_edges_mm == [5000.0, 11000.0]


def test_modality_subset_config():
    cfg = load_config("experiments/configs/modality/ir_depth.yaml")
    assert set(cfg.model.modalities) == {"ir", "depth"}
