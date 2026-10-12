# Reproducing the paper results

Hardware in the paper: **1 × NVIDIA RTX 4090**, PyTorch 2.1 + Ultralytics 8.1,
300 epochs × batch 16 × 640² × 3 seeds {2026, 3407, 7102}. A 24 GB card
(GTX 3090 / RTX 4080) works with batch 8 + gradient accumulation. CPU-only
runs the smoke pipeline (synthetic data) in minutes.

## 0. Setup

```bash
git clone https://github.com/gejingjun/Tri-Net-urban-target-detect.git
cd Tri-Net-urban-target-detect
conda env create -f environment.yml && conda activate tfgnet   # or:
pip install -r requirements.txt && pip install -e .
python scripts/download_checkpoints.py        # optional COCO init for RGB branch
python scripts/make_fake_data.py --out data/fake --n 24        # smoke data
pytest -m "not slow"                          # all unit tests pass on CPU
```

## 1. Smoke run (synthetic, ~1 min)

```bash
python scripts/train.py --config experiments/configs/smoke_fake.yaml --seed 2026
python scripts/eval.py --config experiments/configs/smoke_fake.yaml \
    --ckpt runs/smoke_fake_seed2026/best.pt
```

## 2. Full reproduction (needs data/urbantraffic3m, see data/README.md)

One command per table — `scripts/run_all_experiments.py --set <name>`:

| Table | Command | Paper target |
|---|---|---|
| Main (ours) | `--set paper` (or `train.py --config experiments/configs/tfgnet_urbantraffic3m.yaml` × 3 seeds) | 71.3 ±0.3 mAP@50, 41.8 ±0.2 mAP@50:95, 20.6 M / 52.0 G / 27 FPS |
| Tab.6 modality | `--set modality` | RGB 58.2 / IR 44.7 / D 28.3 / RGB+IR 64.1 / ours 71.3 |
| Tab.7 SOTA | `--set sota` | early 63.2 / late 65.8 / DEGF 64.1 / w/o VADE 66.7 |
| Tab.9 ablation | `--set ablation` | +1.7 / +3.1 / +1.2 / +1.2 → 71.3 |
| Tab.10 VADE phases | `--set vade` | 66.7 / 67.8 / 68.2 / **71.3** |
| Tab.8 encoding | `--set encoding` | raw 63.2 / nmf(10,30) 64.1 / nmf(5,11) 66.8 / inv+valid 67.1 / PhaseA 68.9 |
| Tab.11 conditions | `eval.py --group-by condition` on test | fog 56.8 vs RGB 38.4 |
| Tab.12 batches | `eval.py --group-by batch` | ours 71.8/70.4/71.3/69.7/72.1 |
| Tab.4.5.2 degradation | `eval.py --degradation depth_mask:0.3 \| depth_mask:0.7 \| depth_remove \| ir_remove \| rgb_remove \| ir_mask:0.5` + `sota/ours_wo_vade_gating.yaml` | 70.1 / 67.8 / 65.4 / 62.1 / 49.8 / 58.3; w/o VADE-B 61.2 |
| Tab.13 DroneVehicle | `scripts/make_sgm_depth.py` then `--set cross_dataset` (zero-shot from UrbanTraffic-3M ckpt) | 65.4 tri-modal, +4.4 over DEGF |
| Tab.14 RGBDT500 | `scripts/convert_rgbdt500.py` then `--set cross_dataset` | 71.9 mAP@50 |
| Efficiency | `scripts/bench.py` | 20.6M / 52.0G / 37.0 ms / 27 FPS |

Aggregate summaries land in `experiments/logs/*.json` (mean±std over the 3
paper seeds, exactly as the tables report).

## 3. Docker (one-command environment parity)

```bash
docker compose build
docker compose run --rm tfgnet python scripts/run_all_experiments.py --set paper
```

## 4. Figures (Grad-CAM / validity maps, Sec. 4.6)

```bash
python scripts/visualize.py --config experiments/configs/tfgnet_urbantraffic3m.yaml \
    --ckpt runs/tfgnet_full_seed2026/best.pt --images data/urbantraffic3m/images/val \
    --out results/figures
```

## 5. What the numbers depend on

- `yolov8s.pt` present or not changes the RGB-branch init (±~1.5 mAP).
- Quantile thresholds τ (Eq. 11) are refit on the *train* split by
  `scripts/train.py` automatically; the paper's measured values
  {4702, 7556, 10848} mm are the fallback when a split cannot be fitted.
- `--set quick` exercises every table in seconds on synthetic data but its
  metrics are meaningless — never quote them.
