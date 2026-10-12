"""End-to-end tests: full TFGNet forward/backward on synthetic data,
plus the training and evaluation pipelines (Algorithm 1)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfgnet.config import Config, load_config
from tfgnet.model import TFGNet, build_model


def _cfg(fake_root):
    cfg = Config()
    cfg.data.root = fake_root
    cfg.train.batch = 2
    cfg.train.workers = 0
    cfg.train.device = "cpu"
    cfg.train.amp = False
    cfg.model.rgb_pretrained = None
    return cfg


def test_full_forward_backward():
    cfg = _cfg("data/fake")
    Path("data/fake/images/train").mkdir(parents=True, exist_ok=True)
    if not list(Path("data/fake/images/train").glob("*.png")):
        pytest.skip("run scripts/make_fake_data.py first")
    model = TFGNet(cfg)
    x = torch.rand(1, 3, 640, 640)
    e = torch.rand(1, 6, 640, 640)
    det, aux = model(x, x.clone(), e)
    assert len(det) == 3
    # gradient flows to every module family
    loss = sum(o.sum() for o, r in det)
    loss.backward()
    for name in ("rgb_bb", "ir_bb", "dep_bb", "sdfe", "vade_gate", "msgdff",
                 "rugate", "neck", "head"):
        p = getattr(model, name)
        mod = p[0] if isinstance(p, torch.nn.ModuleList) else p
        assert any(q.grad is not None for q in mod.parameters()), name


@pytest.mark.slow
def test_smoke_training_one_iter(fake_root):
    """One full Trainer iteration + eval on synthetic data."""
    from tfgnet.train import Trainer
    cfg = _cfg(fake_root)
    tr = Trainer(cfg, run_name="pytest")
    batch = next(iter(tr.loader()))
    loss, parts = tr.train_step(batch)
    assert torch.isfinite(loss) and float(loss) > 0
    res = tr.evaluate()
    assert "mAP50" in res


@pytest.mark.slow
def test_degradation_zero_rgb(fake_root):
    """RGB-removed model still predicts (graceful degradation, Sec. 4.5.2)."""
    cfg = _cfg(fake_root)
    cfg.degradation = {"rgb_remove": 1}
    model = build_model(cfg).eval()
    x = torch.rand(1, 3, 640, 640)
    e = torch.rand(1, 6, 640, 640)
    out = model.predict(x, x.clone(), e, conf=0.01)
    assert len(out) == 1


@pytest.mark.parametrize("nm", ["rgb_only", "ir_only", "depth_only", "rgb_ir",
                                "rgb_depth", "ir_depth", "ours"])
def test_modality_subset_configs_load(nm):
    cfg = load_config(f"experiments/configs/modality/{nm}.yaml")
    m = build_model(cfg)
    assert isinstance(m, TFGNet)


def test_parameter_budget():
    """Paper: total 20.6M params (RGB 9.4 + IR 9.4 + depth 0.2 + fusion 1.6).
    We assert the *structure* keeps this budget within tolerance."""
    cfg = Config()
    cfg.model.rgb_pretrained = None
    model = TFGNet(cfg)
    def cnt(mod):
        return sum(p.numel() for p in mod.parameters()) / 1e6
    total = cnt(model)
    assert 12.0 < total < 30.0, total
    assert abs(cnt(model.rgb_bb) - cnt(model.ir_bb)) < 1e-6
