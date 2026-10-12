# API reference (`tfgnet`)

Public surface of the package. All entry points accept/return `torch.Tensor`.

## `tfgnet.config`

```python
from tfgnet.config import Config, load_config
cfg = load_config("experiments/configs/tfgnet_urbantraffic3m.yaml")  # -> Config
```

| Class | Key fields |
|---|---|
| `ModelConfig` | `use_sdfe/use_vade_a/use_vade_b/use_rugate/use_depth`, `modalities`, `naive_early_fusion`, `late_fusion`, `degf_style`, `depth_encoding_mode`, `c_d`, `quantiles`, `region_k`, `p_drop`, `theta_low/high/diff`, `n_scenes`, `rgb_pretrained` |
| `DataConfig` | `root`, `train_split`, `val_split`, `img_size`, `nc`, `names`, `augment`, `stats_json` |
| `TrainConfig` | `epochs=300`, `batch=16`, `lr0=0.01`, `momentum=0.937`, `weight_decay=5e-4`, `warmup_epochs=10`, `lambda_valid=0.1`, `lambda_scene=0.1`, `seeds=[2026,3407,7102]`, `amp`, `save_dir` |

`Config.degradation` (dict) enables the Sec. 4.5.2 eval perturbations:
`{"depth_mask": 0.3}`, `{"depth_remove": 1}`, `{"ir_mask": 0.5}`, `{"ir_remove": 1}`, `{"rgb_remove": 1}`.

## `tfgnet.model`

```python
from tfgnet.model import TFGNet, SingleBranchNet, build_model
model = build_model(cfg)            # dispatches on cfg.model flags
det, aux = model(rgb, ir, e_dep)    # Algorithm 1; rgb/ir (B,3,H,W) in [0,1],
                                    # e_dep (B,Cd,H,W) from VADE Phase A
preds = model.predict(rgb, ir, e_dep, conf=0.25, iou=0.65)
# -> list per batch item of (boxes[N,4] xyxy, scores[N], classes[N])
```

`aux` carries debug/monitoring tensors: `g_pixel` (VADE-B gate, target of
`L_valid`), `scene` (RU-Gate scene posterior), `w_router` (MS-GDFF scale
weights), `m_deg` / `m_con` (RU-Gate mode masks).

## `tfgnet.modules`

| Symbol | Signature | Paper |
|---|---|---|
| `DepthEncoder(c_d=6, d_min=300, d_max=19999, quantiles=(.25,.5,.75), mode='vade_a')` | `fit(valid_depths)`, `encode(depth_HW) -> (C,H,W)` | Eqs. 9-14, Table 8 |
| `SDFE(channels)` | `(x_vis, x_ir) -> (f_vis, f_ir)` | Eqs. 2-8 |
| `VADEGate(c_d=6, region_k=16, p_drop=0.15)` | `(x_dep, e_dep) -> x_dep~` | Eqs. 15-17 |
| `MSGDFF(c_in, c_out)` | `(f_vis, f_ir, f_dep) -> dict(f_fuse, f_shared, w, g_vis, g_ir, g_dep)` | Eqs. 18-23 |
| `RUGate(c, n_scenes=5, theta_low=.3, theta_high=.7, theta_diff=.3)` | `(...) -> (f_out, info)` | Eqs. 24-31 |
| `QGU(c, scale)` | `(x) -> (B,1,H,W)` quality map | Eq. 23 |
| `SceneClassifier(c, K=5)` | `(f_vis) -> (B,K)` softmax | Eq. 29a |
| `AppearanceBackbone(in_ch, w=0.5, d=0.33, pretrained=None)` | `(x) -> (p3,p4,p5)` | Sec. 4.1.2 |
| `DepthBackbone(in_ch=6, chans=(16,32,64))` | `(e_dep) -> (p3,p4,p5)` | Sec. 4.1.2 |
| `Neck(ch, d)` / `DetectHead(ch, nc, reg_max=16)` | YOLOv8 FPN+PAN / decoupled head | Sec. 4.1.2 |

## `tfgnet.data`

```python
from tfgnet.data import TriModalDataset, TriModalAugmentor, batch_stratified_split
ds = TriModalDataset("data/urbantraffic3m", "train", 640, encoder, augmentor, nc=12)
ds.fit_encoder()          # Eq. 11 quantile thresholds from the train split
ds.bind_mosaic_pool()     # mosaic neighbours for TriModalAugmentor
batch = ds[i]             # dict(rgb, ir, e_dep, valid, boxes, cls, n, stem, batch, condition, depth_invalid_frac)
```

## `tfgnet.losses`

```python
from tfgnet.losses import DetectionLoss, AuxiliaryLosses
dl = DetectionLoss(nc=12)                    # loss_box(CIoU)+loss_cls(BCE)+loss_dfl(DFL)
aux = AuxiliaryLosses(0.1, 0.1)
total = dl(...) + aux.validity(g_pixel, valid) + aux.scene(scene, labels)
```

## `tfgnet.train`

```python
from tfgnet.train import Trainer
tr = Trainer(cfg, run_name="tfgnet_full_seed2026")
best_map = tr.fit()                          # saves runs/<name>/{best,last}.pt
```

## `tfgnet.evaluation`

```python
from tfgnet.evaluation import evaluate_with_areas, dump_coco_json
res = evaluate_with_areas(preds, gts)        # mAP50, mAP50_95, per_class, AP_S
dump_coco_json(preds, "preds.json")          # official pycocotools cross-check
```
