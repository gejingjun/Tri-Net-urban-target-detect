"""TFG-Net — complete model per Algorithm 1 (paper Sec. 3.5).

    [VADE-A]  I_dep -> E_dep                       (dataset-side, Eq. 13)
    [Backbone] three branches -> X_m^{P3,P4,P5}
    for each level l:
        [SDFE]    (X_vis, X_ir) -> (F_vis, F_ir)
        [VADE-B]  (X_dep, E_dep) -> X_dep~
        [MS-GDFF] (F_vis, F_ir, X_dep~) -> F_fuse, {G_m}, F_shared
        [RU-Gate] F_fuse -> F_out
    [Neck FPN+PAN] F_out^{P3,P4,P5}
    [Head]         detections

Ablation / baseline toggles reproduce every row of the paper tables:
  modalities subset       -> Modality necessity table (missing modalities
                             are zeroed inputs; same network, paper protocol)
  use_sdfe/use_vade_a/use_vade_b/use_rugate -> Component ablation table
  naive_early_fusion      -> "Early fusion (concat)" row (SingleBranchNet)
  late_fusion             -> "Late fusion (ensemble)" row (scripts/eval.py
                             ensembles three single-branch detectors)
  degf_style              -> DEGF-YOLO row (bimodal, uniform DFE gating,
                             no spatial attention, no depth pathway)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .modules.backbone import AppearanceBackbone, DepthBackbone
from .modules.head import Neck, DetectHead
from .modules.msgdff import MSGDFF
from .modules.rugate import RUGate
from .modules.sdfe import SDFE
from .modules.vade import VADEGate


class TFGNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.levels = 3  # P3, P4, P5
        self.rgb_bb = AppearanceBackbone(3, m.width_mult, m.depth_mult,
                                         pretrained=m.rgb_pretrained if m.rgb_pretrained else None)
        self.ir_bb = AppearanceBackbone(3, m.width_mult, m.depth_mult)
        self.dep_bb = DepthBackbone(m.c_d, m.depth_branch_chans) if m.use_depth else None
        ch = self.rgb_bb.out_ch  # (128,256,512)

        self.sdfe = nn.ModuleList([SDFE(c) for c in ch]) if m.use_sdfe else None
        self.vade_gate = VADEGate(m.c_d, m.region_k, m.p_drop) if m.use_vade_b else None
        self.msgdff = nn.ModuleList([MSGDFF(c, c) for c in ch])
        self.rugate = nn.ModuleList([RUGate(c, m.n_scenes, m.theta_low, m.theta_high, m.theta_diff)
                                     for c in ch]) if m.use_rugate else None
        self.neck = Neck(ch, m.depth_mult)
        self.head = DetectHead(ch, nc=cfg.data.nc)
        # degf_style: replace SDFE with uniform (non-spatial) gating like DFE
        self.degf_style = m.degf_style

    # ------------------------------------------------------------------
    def _degf_gate(self, v, i):
        """DEGF-YOLO DFE reference behaviour: *uniform* (global L2) gating,
        no spatial attention — used only by the degf_style baseline row."""
        xd = (v - i).abs()
        xn = xd / xd.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
        gate = torch.tanh(self.degf_gamma * xn + self.degf_beta)
        return v + v * gate, i + i * gate

    def _apply_degradation(self, rgb, ir, e_dep):
        # Modality subset (Table "Modality necessity"): absent modalities -> 0,
        # applied at training AND eval time (each row is its own model/protocol).
        mods = set(self.cfg.model.modalities or ["rgb", "ir", "depth"])
        if "rgb" not in mods:
            rgb = torch.zeros_like(rgb)
        if "ir" not in mods:
            ir = torch.zeros_like(ir)
        if "depth" not in mods:
            e_dep = torch.zeros_like(e_dep)
        # Systematic degradation (Table 4.5.2) is an EVAL-ONLY perturbation;
        # training-time robustness comes from depth modality dropout (§3.3.2d).
        if self.training:
            return rgb, ir, e_dep
        d = self.cfg.degradation or {}
        if d.get("rgb_remove"):
            rgb = torch.zeros_like(rgb)
        if d.get("ir_remove"):
            ir = torch.zeros_like(ir)
        if d.get("depth_remove"):
            e_dep = torch.zeros_like(e_dep)
        if d.get("depth_mask", 0) > 0:
            keep = (torch.rand(e_dep.shape[0], 1, 1, 1, device=e_dep.device)
                    >= d["depth_mask"]).float()
            e_dep = e_dep * keep
        if d.get("ir_mask", 0) > 0:
            keep = (torch.rand(ir.shape[0], 1, 1, 1, device=ir.device)
                    >= d["ir_mask"]).float()
            ir = ir * keep
        return rgb, ir, e_dep

    def forward(self, rgb, ir, e_dep):
        """rgb/ir (B,3,H,W) normalised 0-1; e_dep (B,Cd,H,W) VADE-A output."""
        rgb, ir, e_dep = self._apply_degradation(rgb, ir, e_dep)
        if not self.cfg.model.use_depth:
            e_dep = torch.zeros_like(e_dep)
        xv = self.rgb_bb(rgb)
        xi = self.ir_bb(ir)
        xd = self.dep_bb(e_dep) if self.dep_bb is not None else \
            tuple(torch.zeros_like(t) for t in xv)

        outs_l, aux = [], {"g_pixel": None, "scene": None,
                           "w_router": [], "m_deg": [], "m_con": []}
        for l, (v, i, d) in enumerate(zip(xv, xi, xd)):
            if self.sdfe is not None:
                if self.degf_style:
                    f_v, f_i = self._degf_gate(v, i)
                else:
                    f_v, f_i = self.sdfe[l](v, i)
            else:
                f_v, f_i = v, i
            x_dg = self.vade_gate(d, e_dep) if self.vade_gate is not None else d
            if self.vade_gate is not None and l == 0:
                aux["g_pixel"] = self.vade_gate.last_g_pixel
            fused = self.msgdff[l](f_v, f_i, x_dg)
            aux["w_router"].append(fused["w"])
            if self.rugate is not None:
                f_out, rg = self.rugate[l](f_v, f_i, x_dg,
                                           fused["f_shared"],
                                           fused["g_vis"], fused["g_ir"], fused["g_dep"])
                aux["scene"] = rg["scene"]
                aux["m_deg"].append(rg["m_deg"])
                aux["m_con"].append(rg["m_con"])
            else:
                f_out = fused["f_fuse"]
            outs_l.append(f_out)
        p3, p4, p5 = self.neck(*outs_l)
        det = self.head([p3, p4, p5])
        return det, aux

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, rgb, ir, e_dep, conf=0.25, iou=0.65):
        self.eval()
        with torch.no_grad():
            det, _ = self.forward(rgb, ir, e_dep)
        from .modules.head import xyxy_nms
        out = []
        for boxes, scores, classes in self.head.decode(det, conf_thres=conf):
            k = xyxy_nms(boxes, scores, classes, iou_thres=iou)
            out.append((boxes[k], scores[k], classes[k]))
        return out


class SingleBranchNet(nn.Module):
    """Plain YOLOv8s-style detector.

    Used for: 'RGB only' / 'IR only' / 'Depth only' (Modality necessity),
    'Early fusion (concat)' (channels = 3+3+Cd), late-fusion members, and
    per-modality DroneVehicle/RGBDT500 YOLOv8s baselines.
    """

    def __init__(self, in_ch=3, nc=12, width_mult=0.5, depth_mult=0.33,
                 pretrained=None):
        super().__init__()
        from .modules.backbone import AppearanceBackbone
        self.bb = AppearanceBackbone(in_ch, width_mult, depth_mult, pretrained)
        self.neck = Neck(self.bb.out_ch, depth_mult)
        self.head = DetectHead(self.bb.out_ch, nc=nc)

    def forward(self, x):
        p3, p4, p5 = self.bb(x)
        return self.head(self.neck(p3, p4, p5)), {}

    @torch.no_grad()
    def predict(self, x, conf=0.25, iou=0.65):
        self.eval()
        det, _ = self.forward(x)
        from .modules.head import xyxy_nms
        out = []
        for boxes, scores, classes in self.head.decode(det, conf_thres=conf):
            k = xyxy_nms(boxes, scores, classes, iou_thres=iou)
            out.append((boxes[k], scores[k], classes[k]))
        return out


def build_model(cfg: Config) -> nn.Module:
    m = cfg.model
    if m.naive_early_fusion:
        # Early fusion: raw RGB(3) + IR(3) + depth-encoded Cd channels
        # concatenated at the input of one branch (Table Main Comparison row).
        return SingleBranchNet(in_ch=3 + 3 + m.c_d, nc=cfg.data.nc,
                               width_mult=m.width_mult, depth_mult=m.depth_mult)
    if m.late_fusion:
        raise ValueError("late_fusion is an ensemble protocol, see scripts/eval.py --late-fusion")
    return TFGNet(cfg)
