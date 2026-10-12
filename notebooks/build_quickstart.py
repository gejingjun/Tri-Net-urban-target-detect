#!/usr/bin/env python3
"""Build notebooks/quickstart.ipynb (demo notebook).

Run: python notebooks/build_quickstart.py
"""
import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


cells = [
    md(["# TFG-Net quickstart demo\n",
        "TFG-Net: Tri-Modal Feature-Guided Fusion with Robust Uncertainty Gating\n",
        "for Urban Object Detection in Degraded Environments\n",
        "([code](https://github.com/gejingjun/Tri-Net-urban-target-detect))\n\n",
        "Runs end-to-end on the synthetic sample data from\n",
        "`python scripts/make_fake_data.py --out data/fake --n 24`.\n",
        "Swap the data root for `data/urbantraffic3m` and load a trained\n",
        "checkpoint for real numbers."]),
    code(["import sys, torch\n",
          "sys.path.insert(0, '../src')\n",
          "from tfgnet.config import load_config\n",
          "from tfgnet.model import build_model\n",
          "from tfgnet.data.dataset import TriModalDataset\n",
          "from tfgnet.modules.vade import DepthEncoder\n",
          "cfg = load_config('../experiments/configs/smoke_fake.yaml')\n",
          "model = build_model(cfg).eval()\n",
          "print('torch', torch.__version__, '| params %.1f M' % (sum(p.numel() for p in model.parameters())/1e6))"]),
    code(["# VADE Phase A: inspect the encoded depth representation (Eqs. 9-13)\n",
          "import matplotlib.pyplot as plt\n",
          "ds = TriModalDataset('../data/fake', 'train', 640, DepthEncoder(), None)\n",
          "b = ds[0]; e = b['e_dep']\n",
          "names = ['valid', 'inv-depth', 'bin1', 'bin2', 'bin3', 'bin4']\n",
          "fig, axes = plt.subplots(1, 6, figsize=(20, 3))\n",
          "for ax, ch, nm in zip(axes, e, names):\n",
          "    ax.imshow(ch, cmap='magma'); ax.set_title(nm); ax.axis('off')\n",
          "plt.show()"]),
    code(["# Forward pass (untrained by default; load a checkpoint for real output)\n",
          "# import torch; ck = torch.load('../runs/tfgnet_full_seed2026/best.pt', map_location='cpu')\n",
          "# model.load_state_dict(ck['model'])\n",
          "rgb, ir, e_dep = b['rgb'][None], b['ir'][None], b['e_dep'][None]\n",
          "boxes, scores, classes = model.predict(rgb, ir, e_dep, conf=0.25)[0]\n",
          "print('detections:', len(boxes))"]),
    code(["# Detections + VADE-B validity gate side by side\n",
          "import numpy as np, cv2\n",
          "det, aux = model(rgb, ir, e_dep)\n",
          "img = (rgb[0].permute(1, 2, 0).numpy()[..., ::-1] * 255).astype('uint8')\n",
          "for (x1, y1, x2, y2), c, s in zip(boxes.numpy(), classes.numpy(), scores.numpy()):\n",
          "    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),\n",
          "                    ((20 + c*17) % 255, 255, 128), 2)\n",
          "    cv2.putText(img, cfg.data.names[int(c)] + ' %.2f' % s,\n",
          "                (int(x1), max(int(y1) - 4, 12)), 0, 0.5, (255, 255, 255))\n",
          "plt.figure(figsize=(12, 6))\n",
          "plt.subplot(1, 2, 1); plt.imshow(img); plt.title('detections'); plt.axis('off')\n",
          "plt.subplot(1, 2, 2); plt.imshow(aux['g_pixel'][0, 0].numpy(), cmap='magma')\n",
          "plt.title('VADE-B pixel validity gate'); plt.axis('off')\n",
          "plt.tight_layout(); plt.show()"]),
    md(["**Next steps**: train on the real data\n",
        "`python scripts/train.py --config experiments/configs/tfgnet_urbantraffic3m.yaml`\n",
        "(300 epochs x 3 seeds), reproduce every table with\n",
        "`python scripts/run_all_experiments.py --set paper`,\n",
        "and read `docs/REPRODUCE.md`."]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).with_name("quickstart.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
