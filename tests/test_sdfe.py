"""SDFE (Sec. 3.2, Eqs. 2-8) unit tests."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfgnet.modules.sdfe import SDFE


def test_shapes_and_residual_identity():
    sdfe = SDFE(8)
    x_vis = torch.randn(2, 8, 16, 16)
    x_ir = torch.randn(2, 8, 16, 16)
    f_v, f_i = sdfe(x_vis, x_ir)
    assert f_v.shape == x_vis.shape and f_i.shape == x_ir.shape
    # force A_spatial = 0 -> Eq.8 collapses to pure residual (F = X)
    class _Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3],
                               device=x.device)
    sdfe.attn = _Zero()
    f0_v, f0_i = sdfe(x_vis, x_ir)
    assert torch.allclose(f0_v, x_vis, atol=1e-6)
    assert torch.allclose(f0_i, x_ir, atol=1e-6)


def test_zero_difference_gives_no_enhancement():
    """X_vis == X_ir -> X_diff=0 -> attention small; enhancement vanishes
    where modalities agree (vegetation/background, paper Sec. 3.2 physical
    interpretation)."""
    sdfe = SDFE(4)
    x = torch.ones(1, 4, 8, 8)
    f_v, f_i = sdfe(x, x.clone())
    diff = (f_v - x).abs().mean()
    assert diff < 1e-3


def test_divergent_regions_enhanced_more():
    """Eq.3 with a monotone positive attention kernel: divergent region gets
    higher A_spatial; and where RGB==IR (Eq.2 X_diff=0 -> gate tanh(beta)=0
    at init) the residual enhancement is exactly zero (Sec. 3.2 physical
    interpretation: agreeing background is suppressed)."""
    sdfe = SDFE(4).eval()
    with torch.no_grad():
        sdfe.dw.weight.fill_(1.0 / 25.0)           # 5x5 local mean of |diff|
        sdfe.attn[0].weight.fill_(1.0)             # 3x3 sum over channels
    x = torch.ones(1, 4, 16, 16)
    y = x.clone()
    y[..., 4:8, 4:8] += 5.0                         # one divergent region
    f_v, _ = sdfe(x, y)
    enh = (f_v - x).abs()
    assert enh[:, :, 5:7, 5:7].mean() > 1e-3       # target enhanced
    assert enh[:, :, 12:16, 12:16].mean() < 1e-8   # agreeing far background: 0
