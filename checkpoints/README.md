# Pretrained checkpoints

Large weights are **not** committed to git (see `.gitignore`); place the
following files in this directory:

| File | Purpose | Where to get it |
|---|---|---|
| `yolov8s.pt` | COCO pre-training for the RGB branch (paper Sec. 4.1.2: "RGB branch initialized from COCO pre-training") | `pip install ultralytics && yolo download yolov8s.pt` — or https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt |
| `tfgnet_urbantraffic3m.pt` | TFG-Net trained on UrbanTraffic-3M (71.3 mAP@50) | HuggingFace: https://huggingface.co/gejingjun/tfgnet (once released); also mirrored with the dataset package (Baidu Pan, code `xww8`) |
| `tfgnet_rgbdt500.pt` | TFG-Net trained on RGBDT500 (71.9 mAP@50) | same as above |

Auto-download helper (after placing `yolov8s.pt` only):

```bash
python scripts/download_checkpoints.py            # fetch yolov8s.pt
```

If `yolov8s.pt` is absent, training gracefully starts from scratch with a
warning — the RGB-branch initialisation adds ~1.5 mAP (paper ablation
context) but is not required for pipeline reproduction.

## Git LFS

To add large weights to the repo instead:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes checkpoints/tfgnet_urbantraffic3m.pt
```
