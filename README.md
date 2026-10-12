# TFG-Net — Tri-Modal Feature-Guided Fusion with Robust Uncertainty Gating for Urban Object Detection in Degraded Environments

**Official PyTorch implementation.**
Code: [https://github.com/gejingjun/Tri-Net-urban-target-detect](https://github.com/gejingjun/Tri-Net-urban-target-detect)
Paper: *TFG-Net: Tri-Modal Feature-Guided Fusion with Robust Uncertainty
Gating for Urban Object Detection in Degraded Environments*, Pattern
Recognition (submitted, revision 2026-10-11).

RGB + Infrared + Depth fusion for urban traffic detection. TFG-Net addresses
three failure modes of existing multimodal fusion — (1) spatially uniform
enhancement, (2) structurally absent depth (**28.3 %** of pixels invalid),
(3) fusion collapse when several sensors degrade at once — with three
reliability-aware modules at defined mounting points:

| Module | Mounting point | Fixes | Code |
|---|---|---|---|
| **SDFE** (Spatial-Differential Feature Enhancement) | post-backbone | spatial uniformity | [`src/tfgnet/modules/sdfe.py`](src/tfgnet/modules/sdfe.py) |
| **VADE** (Validity-Aware Depth Encoding & Fusion, Phase A + B) | preprocessing + fusion | depth invalidity | [`src/tfgnet/modules/vade.py`](src/tfgnet/modules/vade.py) |
| **MS-GDFF** (scale-aware tri-modal routing) | fusion | heterogeneous tri-modal fusion | [`src/tfgnet/modules/msgdff.py`](src/tfgnet/modules/msgdff.py) |
| **RU-Gate** (robust uncertainty gate, 3 modes) | post-fusion | environmental unawareness | [`src/tfgnet/modules/rugate.py`](src/tfgnet/modules/rugate.py) |

Built on a YOLOv8s three-branch backbone (RGB / IR / lightweight depth CNN)
with an FPN+PAN neck and decoupled anchor-free head. **20.6 M params,
52.0 GFLOPs, 27 FPS** on a single RTX 4090.

## Results (paper headline numbers)

| Benchmark | mAP@50 | mAP@50:95 |
|---|---|---|
| UrbanTraffic-3M (3,000 triplets, 12 cls) | **71.3 ± 0.3** (+5.5 over strongest tri-modal baseline) | 41.8 ± 0.2 |
| RGBDT500 (203,700 triplets, ≥66 cls) | **71.9** | — |
| DroneVehicle (zero-shot transfer, SGM depth) | **65.4** | — |

Per-modality necessity, ablations (SDFE +1.7 / VADE +3.1 / RU-Gate +1.2 /
full-depth +1.2 → 71.3), VADE phase decomposition, depth-encoding comparison,
systematic degradation, condition/batch breakdowns and efficiency are all
reproducible from this repository — see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Installation

Requirements: Python ≥ 3.9, CUDA ≥ 11.8 (training), one NVIDIA GPU with
≥ 16 GB (paper: 1 × RTX 4090). CPU works for smoke tests and inference.

```bash
git clone https://github.com/gejingjun/Tri-Net-urban-target-detect.git
cd Tri-Net-urban-target-detect
conda env create -f environment.yml && conda activate tfgnet
# or
pip install -r requirements.txt && pip install -e .
python scripts/download_checkpoints.py          # optional COCO init (yolov8s.pt)
```

Docker one-command environment:

```bash
docker compose build && docker compose run --rm tfgnet bash
```

## Quickstart

```bash
# 1. synthetic sample data (no download needed)
python scripts/make_fake_data.py --out data/fake --n 24
# 2. train the full model for 2 epochs (sanity run)
python scripts/train.py --config experiments/configs/smoke_fake.yaml --seed 2026
# 3. evaluate
python scripts/eval.py --config experiments/configs/smoke_fake.yaml \
    --ckpt runs/smoke_fake_seed2026/best.pt
# 4. demo notebook: run notebooks/build_quickstart.py then open notebooks/quickstart.ipynb
```

Reproduce a paper table (real data):

```bash
python scripts/run_all_experiments.py --set ablation       # Table 9
python scripts/run_all_experiments.py --set paper          # everything
```

## Data

| Dataset | Size | Get |
|---|---|---|
| **UrbanTraffic-3M** (new) | 3,000 RGB-IR-Depth triplets, 12 classes | Baidu Pan <https://pan.baidu.com/s/17luVd99FGrdL4W5LgMvBKg?pwd=xww8> (code `xww8`) |
| **RGBDT500** (public) | 203,700 triplets, ≥66 classes | official RGBDT release; Baidu mirror code `j3ek` |
| **DroneVehicle** (public) | 26,970 RGB-IR pairs, 5 classes | <https://github.com/zhaoxiangyu97/DroneVehicle> + SGM depth via `scripts/make_sgm_depth.py` |

Place extracted folders under `data/` in the layout documented in
[`data/README.md`](data/README.md) and verify with
`python scripts/check_dataset.py --root data/urbantraffic3m` (checks the
28.3 % invalid-depth and 15.5 % pseudo-depth statistics from the paper).

## Model weights

Pretrained checkpoints: `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md) — yolov8s COCO init via
`scripts/download_checkpoints.py`; TFG-Net weights released with the paper).
Large files are Git-LFS tracked.

## Repository structure

```
├── src/tfgnet/          # package: modules, model, data, losses, train, eval
│   ├── modules/         # sdfe · vade · msgdff · rugate · backbone · head · common
│   ├── data/            # dataset · augment (mosaic/mixup/hflip/HSV) · splits
├── scripts/             # train · eval · bench · visualize · run_all_experiments · data prep
├── experiments/configs/ # 36 YAMLs = every paper table row (Tab. 6-14)
├── tests/               # formula-level + pipeline unit tests (pytest)
├── notebooks/           # quickstart.ipynb demo
├── docs/                # ARCHITECTURE (code↔paper equations) · REPRODUCE · API
├── data/  checkpoints/  results/  runs/
├── requirements.txt  environment.yml  pyproject.toml
├── Dockerfile  docker-compose.yml  .gitignore  LICENSE  CITATION.cff  CONTRIBUTING.md
```

## Citation

```bibtex
@article{tfgnet2026,
  title   = {TFG-Net: Tri-Modal Feature-Guided Fusion with Robust Uncertainty
             Gating for Urban Object Detection in Degraded Environments},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Code: https://github.com/gejingjun/Tri-Net-urban-target-detect}
}
```

## License

MIT — see [`LICENSE`](LICENSE). © 2026 the TFG-Net contributors.
