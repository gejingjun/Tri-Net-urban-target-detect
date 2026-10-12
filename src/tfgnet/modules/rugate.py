"""RU-Gate — Robust Uncertainty Gate (paper Sec. 3.5).

Mounting point: post-fusion (MS-GDFF), pre-neck. Implements Eqs. 24-30:

    Eq. 24  U_m       = sigmoid(lambda_u * Var_{5x5}(G_m))
    Eq. 25  G~_m      = (1 - U_m) * G_m              (uncertainty-modulated)
    Eq. 26  M_deg     = prod_m 1[G~_m < theta_low]
    Eq. 27  M_con     = prod_m 1[G~_m > theta_high] * 1[max|G~_m-G~_m'|>theta_diff]
    Eq. 28  F_normal  = sum_m G_m * F_shared / sum_m G_m
          Scene cls:  s = softmax(MLP(GAP(F_vis)))            (Eq. 29a)
          Prior:      w_prior = softmax(s^T W_prior)          (Eq. 29b)
          F_robust    = sum_m w_prior_m * F_m
          F_negotiate = min_m(G_m)*F_shared + max_m(G_m)*Fbar_best (Eq. 30)
    Eq. 31  F_out = (1-M_deg-M_con)*F_normal + M_deg*F_robust + M_con*F_negotiate
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Scene types (paper Sec. 3.5, example enumeration)
SCENE_TYPES = ("daytime-clear", "nighttime", "fog", "glare", "backlight")


class LocalVar(nn.Module):
    """Var_{5x5} via E[x^2] - E[x]^2 (Eq. 24)."""

    def __init__(self, k=5):
        super().__init__()
        self.avg = nn.AvgPool2d(k, stride=1, padding=k // 2, count_include_pad=False)

    def forward(self, x):
        mu = self.avg(x)
        mu2 = self.avg(x * x)
        return (mu2 - mu * mu).clamp(min=0.0)


class SceneClassifier(nn.Module):
    """GAP -> MLP(ReLU) -> softmax over K scene types (Eq. 29a)."""

    def __init__(self, c, n_scenes=len(SCENE_TYPES), hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_scenes))

    def forward(self, f_vis):
        return torch.softmax(self.mlp(f_vis.mean((2, 3))), dim=-1)     # (B,K)


class RUGate(nn.Module):
    def __init__(self, c, n_scenes=len(SCENE_TYPES),
                 theta_low=0.1, theta_high=0.25, theta_diff=0.15,
                 lambda_u_init=5.0):
        # NOTE on thresholds: G~_m = (1-U_m)*G_m with U_m = sigmoid(lambda_u*Var)
        # For spatially flat quality maps U -> 0.5, so G~ tops out at 0.5 and
        # the defaults are scaled accordingly (degradation: G~ < 0.1 ~ G < 0.2;
        # conflict: all G~ > 0.25 ~ G > 0.5 with pairwise spread > 0.15).
        super().__init__()
        self.var = LocalVar(5)
        self.lambda_u = nn.Parameter(torch.tensor(float(lambda_u_init)))
        self.theta_low, self.theta_high, self.theta_diff = theta_low, theta_high, theta_diff
        # Eq. 29a
        self.scene = SceneClassifier(c, n_scenes)
        # Eq. 29b: learnable prior lookup W_prior in R^{K x M} (M=3 modalities).
        # Initialised near-uniform; offline per-scene statistics can be loaded
        # via set_prior_weights() before fine-tuning end-to-end.
        self.w_prior = nn.Parameter(torch.zeros(n_scenes, 3))
        self.last_scene_logits: torch.Tensor | None = None  # for L_scene supervision

    def set_prior_weights(self, wt: torch.Tensor):
        with torch.no_grad():
            self.w_prior.copy_(wt)

    def forward(self, f_vis, f_ir, f_dep, f_shared, g_vis, g_ir, g_dep):
        g = torch.stack([g_vis, g_ir, g_dep], dim=1)                  # (B,3,1?,H,W)-> squeeze
        g = g.squeeze(2) if g.shape[2] == 1 else g                    # (B,3,H,W)
        f_m = torch.stack([f_vis, f_ir, f_dep], dim=1)                # (B,3,C,H,W)
        # Eq. 24-25
        u = torch.sigmoid(self.lambda_u * self.var(g))                # (B,3,H,W)
        g_comp = (1.0 - u) * g
        # Eq. 26: all three degraded
        m_deg = (g_comp < self.theta_low).float().prod(dim=1, keepdim=True)
        # Eq. 27: all high but pairwise-disagreeing
        all_high = (g_comp > self.theta_high).float().prod(dim=1, keepdim=True)
        pairwise = g_comp.unsqueeze(1) - g_comp.unsqueeze(2)           # (B,3,3,H,W)
        max_diff = pairwise.abs().amax(dim=(1, 2))                     # (B,H,W)
        conflict = (max_diff > self.theta_diff).float().unsqueeze(1)   # (B,1,H,W)
        m_con = all_high * conflict                                   # (B,1,H,W)
        # Mode 1 (Eq. 28) normal cross-gating
        f_normal = (g.unsqueeze(2) * f_shared.unsqueeze(1)).sum(1) / g.unsqueeze(2).sum(1).clamp(min=1e-6)
        # Mode 2 (Eq. 29) scene-prior-weighted robust fusion
        s = self.scene(f_vis)
        self.last_scene_logits = s
        w_prior = torch.softmax(s @ self.w_prior, dim=-1)             # (B,3)
        f_robust = (w_prior[:, :, None, None, None] * f_m).sum(1)
        # Mode 3 (Eq. 30) negotiation / intersection
        g_min = g.amin(dim=1, keepdim=True)                            # (B,1,H,W)
        g_max_idx = g.argmax(dim=1)                                    # (B,H,W)
        idx = g_max_idx.unsqueeze(1).expand(-1, f_m.shape[2], -1, -1)   # (B,C,H,W)
        # f_m: (B,3,C,H,W) -> (B,C,H,W,3); gather max-quality modality per pixel
        fm_last = f_m.permute(0, 2, 3, 4, 1)
        best = fm_last.gather(4, idx.unsqueeze(-1)).squeeze(-1)         # (B,C,H,W)
        f_bar_best = best - f_shared                                  # modality-specific part
        g_max = g.amax(dim=1, keepdim=True)
        f_negotiate = g_min * f_shared + g_max * f_bar_best
        # Eq. 31 final blend
        m_deg = m_deg.clamp(max=1.0)
        m_con = (m_con * (1.0 - m_deg)).clamp(max=1.0)                # deg has precedence
        w_norm = (1.0 - m_deg - m_con).clamp(min=0.0)
        out = w_norm * f_normal + m_deg * f_robust + m_con * f_negotiate
        return out, dict(scene=s, w_prior=w_prior,
                         m_deg=m_deg, m_con=m_con, f_normal=f_normal,
                         f_robust=f_robust, f_negotiate=f_negotiate)
