"""VADE (Sec. 3.3) unit tests — Phase A encoding numerics + Phase B gating."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.modules.vade import DepthEncoder, VADEGate


def test_phase_a_channels_and_validity():
    enc = DepthEncoder()
    d = torch.zeros(32, 32)
    d[4:8, 4:8] = 5000.0          # valid, bin-2-ish
    d[:2, :2] = 0.0               # invalid
    e = enc.encode(d)
    assert e.shape == (6, 32, 32)
    assert e[0].sum() == (d > 0).float().sum()       # Eq.9 validity
    # inverse depth Eq.10: 1/clip(d,300,19999)*19999 -> smaller when farther
    assert e[1, 5, 5] > 0
    # bin channels sum to exactly 1 where valid (bins partition (d_min, d_max])
    # for the valid pixel
    px = e[:, 5, 5]
    # Eq.12 fractional ramp: exactly one bin active, value = (D - lo) / (hi - lo)
    edges = [enc.d_min] + list(enc.tau) + [enc.d_max]
    active = (px[2:] > 0).nonzero().flatten()
    assert len(active) == 1
    bi = int(active[0])
    lo, hi = edges[bi], edges[bi + 1]
    expect = (5000.0 - lo) / (hi - lo)
    assert abs(float(px[2:][bi]) - expect) < 1e-4


def test_phase_a_pseudo_depth_routing():
    """Eq.14: uint8 pseudo-depth (max <= 255) -> [D/255, 1[D>0], zeros...]."""
    enc = DepthEncoder()
    d = torch.randint(0, 256, (16, 16), dtype=torch.float32)
    e = enc.encode(d)
    assert e.shape[0] == 6
    assert torch.allclose(e[0], d / 255.0)
    assert torch.allclose(e[1], (d > 0).float())
    assert e[2:].abs().sum() == 0


def test_phase_a_fit_quantiles():
    enc = DepthEncoder()
    v = torch.arange(1000, 20000, 1000, dtype=torch.float32)
    enc.fit(v)
    assert abs(enc.tau[0] - torch.quantile(v, 0.25)) < 1e-3


def test_phase_b_gate_range_and_dropout():
    gate = VADEGate()
    x_dep = torch.randn(2, 4, 32, 32)
    e = torch.zeros(2, 6, 32, 32)
    e[:, 0, :16] = 1.0            # valid top half
    out = gate(x_dep, e)
    assert out.shape == x_dep.shape
    # invalid bottom half should be suppressed to (near) zero
    assert out[:, :, 16:].abs().max() < x_dep.abs().max()
    # dropout: with p=1 the encoding is zeroed in training mode
    gate.p_drop = 1.0
    gate.train()
    out_dropped = gate(x_dep, e)
    assert torch.allclose(out_dropped, torch.zeros_like(x_dep), atol=1e-6)


def test_other_encoding_modes_table8():
    d = torch.full((8, 8), 12000.0)
    assert DepthEncoder(mode="raw").encode(d).shape == (1, 8, 8)
    assert DepthEncoder(mode="near_mid_far",
                        fixed_edges=[10000, 30000]).encode(d).shape == (3, 8, 8)
    assert DepthEncoder(mode="inv_valid").encode(d).shape == (2, 8, 8)
    assert DepthEncoder().encode(d).shape == (6, 8, 8)
