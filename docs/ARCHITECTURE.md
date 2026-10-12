# TFG-Net architecture: code ↔ paper mapping

Paper: **TFG-Net: Tri-Modal Feature-Guided Fusion with Robust Uncertainty
Gating for Urban Object Detection in Degraded Environments** (Pattern
Recognition, revision 2026-10-11).
Code: <https://github.com/gejingjun/Tri-Net-urban-target-detect>

Algorithm 1 is implemented in [`src/tfgnet/model.py`](../src/tfgnet/model.py)
(`TFGNet.forward`). Every module file names the equation it implements.

| Paper | Where | Code | Notes |
|---|---|---|---|
| Pipeline Eq. 1 (Algorithm 1) | Sec. 3.1 | `src/tfgnet/model.py:TFGNet.forward` | strict feed-forward order |
| Problem formulation (I_vis, I_ir, I_dep, D mm, 0=invalid) | Sec. 3.1 | `src/tfgnet/data/dataset.py:TriModalDataset` | depth loaded uint16 mm |
| **VADE Phase A** Eqs. 9-14 (zero-parameter) | Sec. 3.3.1 | `src/tfgnet/modules/vade.py:DepthEncoder` | `fit()` = Eq.11 quantiles; `encode()` = Eqs. 9-13; `mode=` switches reproduce Table 8 (raw / near_mid_far / inv_valid / vade_a); Eq.14 uint8 pseudo routing built-in |
| **SDFE** Eqs. 2-8 | Sec. 3.2 | `src/tfgnet/modules/sdfe.py:SDFE` | `StdPool2d` = Eq.5; α,γ,β learnable per-channel |
| **VADE Phase B** Eqs. 15-17 + dropout §3.3.2(d) | Sec. 3.3.2 | `src/tfgnet/modules/vade.py:VADEGate` | Conv1×1 init ≈ validity indicator; `p_drop=0.15` removes depth contribution entirely for dropped samples (appearance-only fallback) |
| **MS-GDFF** Eqs. 18-22 | Sec. 3.4 | `src/tfgnet/modules/msgdff.py:MSGDFF` | `ScaleRouter` = Eq.18-19; C3k2 = YOLO11 (`common.py`); `RCCA` = CCNet criss-cross (`common.py`); QGU ×3 + `f_shared` exposed for RU-Gate |
| **RU-Gate** Eqs. 24-31 | Sec. 3.5 | `src/tfgnet/modules/rugate.py:RUGate` | `LocalVar` = Eq.24; masks = Eqs.26-27; Mode1 Eq.28; scene cls Eq.29a (`SceneClassifier`), prior lookup Eq.29b (`w_prior`, `set_prior_weights` loads offline stats); Mode3 Eq.30 (min/max quality negotiation); blend Eq.31 |
| Backbones (CSPDarknet YOLOv8s widths [32,64,128,256,512]; depth 3-layer CNN [16,32,64]) | Sec. 4.1.2 | `src/tfgnet/modules/backbone.py` | `AppearanceBackbone.load_coco_pretrained` remaps yolov8s COCO weights onto the RGB branch |
| Neck FPN+PAN, decoupled anchor-free head | Sec. 4.1.2 | `src/tfgnet/modules/head.py:Neck, DetectHead` | DFL + TAL assignment protocol as YOLOv8 |
| Loss Eq. 32 (CIoU+BCE+DFL + 0.1·L_valid + 0.1·L_scene) | Sec. 3.6 | `src/tfgnet/losses.py` | `TaskAlignedAssigner`, `AuxiliaryLosses.validity/scene`; **no** depth-reconstruction/contrastive loss (by design, Sec. 3.6) |
| Training (SGD 0.01, mom 0.937, wd 5e-4, cosine, 300ep, warmup 10, bs16, 640², Mosaic+MixUp+hflip+HSV, seeds {2026,3407,7102}) | Sec. 3.6 | `src/tfgnet/train.py:Trainer`, `scripts/train.py` | Mosaic applied identically to all three modalities (`src/tfgnet/data/augment.py`) |
| Evaluation mAP@50, mAP@50:95, per-class AP, AP_S (COCO) | Sec. 4.1.3 | `src/tfgnet/evaluation.py` | 101-point interpolation; `scripts/eval.py` also dumps COCO-format predictions |
| Split protocol (2000/1000 batch-stratified, sequence-grouped) | Sec. 4.1.1 / Table 4 | `src/tfgnet/data/splits.py` | `scripts/check_dataset.py` verifies batch counts vs Table 4 |

## Reproducing every table (see docs/REPRODUCE.md for exact commands)

| Paper table | Configs / command |
|---|---|
| Tab.6 Modality necessity | `experiments/configs/modality/*.yaml` (7 rows; absent modalities zeroed in `model.py:_apply_degradation`) |
| Tab.7 Main comparison | `experiments/configs/sota/*.yaml` + `tfgnet_urbantraffic3m.yaml`; `naive_early_fusion`/`late_fusion` flags |
| Tab.9 Component ablation | `experiments/configs/ablation/0*.yaml` (`--set ablation`) |
| Tab.10 VADE phases | `experiments/configs/vade_phase/*.yaml` |
| Tab.8 Depth encoding | `experiments/configs/depth_encoding/*.yaml` (`model.depth_encoding_mode`) |
| Tab.11 Per-condition | `scripts/eval.py --group-by condition` |
| Tab.12 Per-batch | `scripts/eval.py --group-by batch` |
| Tab.4.5.2 Degradation | `scripts/eval.py --degradation depth_mask:0.3 … ir_mask:0.5` |
| Efficiency (params/FLOPs/FPS) | `scripts/bench.py` (thop + warm-up timing, batch 1, 640²) |
| Tab.13 DroneVehicle transfer | `experiments/configs/cross_dataset/dronevehicle_zero_shot.yaml` (zero-shot) + `scripts/make_sgm_depth.py` |
| Tab.14 RGBDT500 | `experiments/configs/cross_dataset/rgbdt500.yaml` + `scripts/convert_rgbdt500.py` |
| Grad-CAM / validity maps (Figs. 8-9) | `scripts/visualize.py` → `results/figures/` |
