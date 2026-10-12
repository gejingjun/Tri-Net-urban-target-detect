"""MS-GDFF (Sec. 3.4, Eqs. 18-22) and RU-Gate (Sec. 3.5, Eqs. 24-30) tests."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.modules.msgdff import MSGDFF, QGU, ScaleRouter
from tfgnet.modules.rugate import RUGate


def test_msgdff_shapes_and_routing_weights():
    m = MSGDFF(4, 4)
    fv = torch.randn(2, 4, 8, 8)
    fi = torch.randn(2, 4, 8, 8)
    fd = torch.randn(2, 4, 8, 8)
    out = m(fv, fi, fd)
    assert out["f_fuse"].shape == (2, 4, 8, 8)
    assert out["f_shared"].shape == (2, 4, 8, 8)
    w = out["w"]                                  # Eq.19 softmax over 3 paths
    assert w.shape == (2, 3)
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)
    for key in ("g_vis", "g_ir", "g_dep"):       # Eq.23 quality maps in [0,1]
        assert out[key].min() >= 0 and out[key].max() <= 1


def test_scale_router_reacts_to_input():
    r = ScaleRouter(12)
    a = r(torch.zeros(1, 12, 8, 8))
    b = r(torch.randn(1, 12, 8, 8))
    assert a.shape == (1, 3)
    assert not torch.allclose(a, b)


def test_rugate_mode_switching():
    """Feed extreme quality maps and check which mode dominates (Eq.31)."""
    c = 4
    rg = RUGate(c)
    fv = torch.randn(1, c, 16, 16)
    fi = torch.randn(1, c, 16, 16)
    fd = torch.randn(1, c, 16, 16)
    fs = torch.randn(1, c, 16, 16)
    # all LOW quality -> degradation mask -> robust mode
    g_lo = torch.full((1, 1, 16, 16), 0.1)
    out, info = rg(fv, fi, fd, fs, g_lo, g_lo, g_lo)
    assert out.shape == fv.shape
    assert info["m_deg"].min() > 0.5
    # all HIGH but wildly disagreeing -> conflict mask -> negotiation
    g_h1, g_h2, g_h3 = (torch.full((1, 1, 16, 16), v) for v in (0.95, 0.98, 0.55))
    out2, info2 = rg(fv, fi, fd, fs, g_h1, g_h2, g_h3)
    assert info2["m_con"].mean() > 0.5
    # normal mid-quality -> normal mode dominates
    g_mid = (torch.full((1, 1, 16, 16), 0.5) + 0.02 * torch.randn(1, 1, 16, 16))
    out3, info3 = rg(fv, fi, fd, fs, g_mid, g_mid, g_mid)
    assert (info3["m_deg"] + info3["m_con"]).max() < 0.1
    w_norm = (1 - info3["m_deg"] - info3["m_con"]).clamp(min=0)
    assert torch.allclose(out3, w_norm * info3["f_normal"], atol=1e-5)


def test_rugate_scene_prior_lookup():
    rg = RUGate(4)
    assert rg.w_prior.shape == (5, 3)              # K x M, Eq.29b
    fv = torch.randn(1, 4, 16, 16)
    g = torch.full((1, 1, 16, 16), 0.1)
    _, info = rg(fv, fv, fv, fv, g, g, g)
    assert info["scene"].shape == (1, 5)
    assert torch.allclose(info["scene"].sum(-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(info["w_prior"].sum(-1), torch.ones(1), atol=1e-5)
