"""Core modules (paper Sec. 3)."""
from .sdfe import SDFE
from .vade import DepthEncoder, VADEGate
from .msgdff import MSGDFF, QGU, ScaleRouter
from .rugate import RUGate, SceneClassifier
from .backbone import AppearanceBackbone, DepthBackbone
from .head import Neck, DetectHead, xyxy_nms

__all__ = ["SDFE", "DepthEncoder", "VADEGate", "MSGDFF", "QGU", "ScaleRouter",
           "RUGate", "SceneClassifier", "AppearanceBackbone", "DepthBackbone",
           "Neck", "DetectHead", "xyxy_nms"]
