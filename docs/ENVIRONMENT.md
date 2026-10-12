# Environment notes

- **Paper environment**: PyTorch 2.1 + CUDA 12.1, Ultralytics 8.1,
  1 × RTX 4090 (24 GB). Training 300 epochs × 3 seeds on UrbanTraffic-3M.
- **This repo's own code paths** are framework-light: modules are pure
  PyTorch (the `ultralytics` pin only matters if you reuse its C3k2/CBS
  weights; `scripts/download_checkpoints.py` + `AppearanceBackbone.load_coco_pretrained`
  handle the name remapping). It has been tested locally on
  Python 3.10-3.14 / torch 2.1-2.13 (CPU and CUDA) — `pytest` passes on CPU.
- Windows: run everything under PowerShell; set `--device cpu` or CUDA_VISIBLE_DEVICES.
- `pip install -e .` exposes the `tfgnet` package; scripts also insert
  `src/` into `sys.path`, so they work without installation.
