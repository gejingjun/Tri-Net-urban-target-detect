"""Backbones (paper Sec. 4.1.2 Implementation Details).

* AppearanceBackbone: YOLOv8s CSPDarknet (5 stages, widths [32,64,128,256,512]).
  RGB and IR branches share the *architecture* but not the weights; the RGB
  branch may be initialised from COCO pre-training (yolov8s.pt).
* DepthBackbone: lightweight 3-layer CNN (3x3 conv, channels [16,32,64],
  BatchNorm + ReLU each) over the Cd=6 VADE-encoded channels, emitting
  P3/P4/P5 features (stride 8/16/32).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .common import C3k2, Conv, SPPF


class AppearanceBackbone(nn.Module):
    """YOLOv8s-style CSPDarknet trunk emitting P3, P4, P5."""

    def __init__(self, in_ch=3, width_mult=0.5, depth_mult=0.33, pretrained=None):
        super().__init__()
        w = lambda c: max(8, int(round(c * width_mult / 8.0)) * 8)
        d = lambda n: max(1, int(round(n * depth_mult)))
        self.stem = Conv(in_ch, w(64), 3, 2)                     # P1/2
        self.s2 = nn.Sequential(                                  # P2/4
            Conv(w(64), w(128), 3, 2), C3k2(w(128), w(128), d(3), True))
        self.s3 = nn.Sequential(                                  # P3/8  (out ch = w(256)=128)
            Conv(w(128), w(256), 3, 2), C3k2(w(256), w(256), d(6), True))
        self.s4 = nn.Sequential(                                  # P4/16 (out ch = w(512)=256)
            Conv(w(256), w(512), 3, 2), C3k2(w(512), w(512), d(6), True))
        self.s5 = nn.Sequential(                                  # P5/32 (ch 512)
            Conv(w(512), w(1024), 3, 2), C3k2(w(1024), w(1024), d(3), False),
            SPPF(w(1024), w(1024), 5))
        self.out_ch = (w(256), w(512), w(1024))                   # (128,256,512) @ s=0.5
        if pretrained:
            self.load_coco_pretrained(pretrained)

    def forward(self, x):
        x = self.stem(x)
        x = self.s2(x)
        p3 = self.s3(x)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return p3, p4, p5

    # ------------------------------------------------------------------
    # Positional mapping from Ultralytics YOLOv8s trunk indices to our trunk.
    # Ultralytics: 0 Conv, 1 Conv, 2 C3k2, 3 Conv, 4 C3k2, 5 Conv, 6 C3k2,
    #              7 Conv, 8 C3k2, 9 SPPF  (10 = final C3k2, not in our trunk)
    _UL_MAP = {
        "stem": "model.0", "s2.0": "model.1", "s2.1": "model.2",
        "s3.0": "model.3", "s3.1": "model.4", "s4.0": "model.5",
        "s4.1": "model.6", "s5.0": "model.7", "s5.1": "model.8",
        "s5.2": "model.9",
    }

    def load_coco_pretrained(self, ckpt_path: str, strict: bool = False):
        """Remap Ultralytics YOLOv8s COCO weights onto this trunk.

        Our Conv subclasses nn.Conv2d directly (``X.weight``) whereas
        Ultralytics wraps conv as ``X.conv.weight``; BN names match.
        """
        from ..utils import get_logger
        log = get_logger()
        if not Path(ckpt_path).exists():
            log.warning("COCO weights %s not found -> training from scratch "
                        "(download: see checkpoints/README.md)", ckpt_path)
            return
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = None
        if isinstance(ck, dict):
            model = ck.get("model", None)
            sd = model.state_dict() if hasattr(model, "state_dict") else ck.get("state_dict", {})
        else:
            sd = ck.state_dict()
        plain = {k: v.float() for k, v in sd.items() if torch.is_tensor(v)}

        def to_ul(name: str) -> str | None:
            for our_pfx, ul_pfx in self._UL_MAP.items():
                if name.startswith(our_pfx + ".") or name == our_pfx:
                    rest = name[len(our_pfx):].lstrip(".")
                    if rest.startswith("bn."):
                        return f"{ul_pfx}.{rest}"
                    if rest in ("weight", "bias"):
                        return f"{ul_pfx}.conv.{rest}"
                    # C3k2 internals: cv1/cv2/m.* keep their names, but the
                    # leaf Conv weight must gain the extra .conv. level
                    if rest.endswith(".weight"):
                        return f"{ul_pfx}.{rest[:-7]}.conv.weight"
                    return f"{ul_pfx}.{rest}"
            return None

        merged, loaded = dict(self.state_dict()), 0
        for k, ours in list(merged.items()):
            ul = to_ul(k)
            if ul and ul in plain and plain[ul].shape == ours.shape:
                merged[k] = plain[ul]
                loaded += 1
        self.load_state_dict(merged, strict=strict)
        log.info("RGB branch COCO init: %d/%d tensors loaded from %s",
                 loaded, len(merged), ckpt_path)


class DepthBackbone(nn.Module):
    """3-layer stride-2 CNN over the Cd-channel VADE encoding (Sec. 4.1.2)."""

    def __init__(self, in_ch=6, chans=(16, 32, 64)):
        super().__init__()
        self.l1 = Conv(in_ch, chans[0], 3, 2)     # stride 2
        self.l2 = Conv(chans[0], chans[1], 3, 2)   # stride 4
        self.l3 = Conv(chans[1], chans[2], 3, 2)   # stride 8 == P3
        # stride-1x1 projections to the fusion widths of the appearance
        # branches (P3=128, P4=256, P5=512); /2 maxpools for P4/P5.
        self.p3 = Conv(chans[2], 128, 1)
        self.p4 = Conv(chans[2], 256, 1)
        self.p5 = Conv(chans[2], 512, 1)
        self.mp = nn.MaxPool2d(2, 2)
        self.out_ch = (128, 256, 512)

    def forward(self, e_dep):
        p3 = self.l3(self.l2(self.l1(e_dep)))
        return self.p3(p3), self.p4(self.mp(p3)), self.p5(self.mp(self.mp(p3)))
