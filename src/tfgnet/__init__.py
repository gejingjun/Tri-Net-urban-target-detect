"""TFG-Net: Tri-Modal Feature-Guided Fusion with Robust Uncertainty Gating
for Urban Object Detection in Degraded Environments.

Reference implementation of the paper (FGF-YOLO / TFG-Net, Pattern Recognition
submission revision 2026-10-11).

Repository: https://github.com/gejingjun/Tri-Net-urban-target-detect

Modules (paper Section 3):
    SDFE      -- Spatial-Differential Feature Enhancement   (Sec. 3.2, Eqs. 2-8)
    VADE-A    -- Range-Aligned Depth Encoding                (Sec. 3.3.1, Eqs. 9-14)
    VADE-B    -- Validity-Gated Depth Fusion                 (Sec. 3.3.2, Eqs. 15-17)
    MS-GDFF   -- Scale-Aware Tri-Modal Routing               (Sec. 3.4, Eqs. 18-22)
    RU-Gate   -- Robust Uncertainty Gate (3 modes)           (Sec. 3.5, Eqs. 23-30)

Every source file referencing an equation carries the paper's equation number
in its docstring; see docs/ARCHITECTURE.md for the full code <-> paper mapping.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .model import TFGNet, build_model

__all__ = ["TFGNet", "build_model", "Config", "load_config", "__version__"]
