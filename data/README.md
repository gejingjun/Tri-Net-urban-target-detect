# Data

Three tri-modal datasets are used in the paper (Sec. 4.1.1). None of them
is committed to git; place the extracted folders under this directory.

## 1. UrbanTraffic-3M (newly collected, 3,000 triplets)

- **Download** (paper-listed): Baidu Pan
  <https://pan.baidu.com/s/17luVd99FGrdL4W5LgMvBKg?pwd=xww8> — extraction code `xww8`
- 12 categories: `person, animal, car, light, sign, bicycle, seat, garbagecan, uav, boat, ball, tricycle`
- Five acquisition batches: `numeric, seq6_3_8, shuming, hehe, seq_suppl`
- Split: 2,000 / 1,000 train/val, batch-stratified and sequence-grouped
  (already shipped as `meta/*.json`; if missing, run
  `python -c "from tfgnet.data.splits import batch_stratified_split; batch_stratified_split(None,'data/urbantraffic3m')"`).

Expected layout (YOLO labels, per-image depth PNG):

```
data/urbantraffic3m/
  images/train/<stem>.png     uint8 RGB, 1920x1080 (resized to 640 by pipeline)
  ir/train/<stem>.png         uint8 IR (effectively grayscale)
  depth/train/<stem>.png      uint16 metric depth in mm (ZED, 0.3-20 m)
                              15.5% of test images are uint8 pseudo-depth —
                              the pipeline routes them via Eq. 14
  labels/train/<stem>.txt     "cls cx cy w h" normalised floats
  meta/train.json             {"<stem>": {"batch": ..., "condition": ...}}
```

Checksums of the extracted tree: `scripts/download_data.sh --verify`
writes `results/urbantraffic3m.sha256`.

## 2. RGBDT500 (public, 203,700 triplets, ≥66 categories)

- Paper-listed mirror (official download link): Baidu Pan
  <https://pan.baidu.com/s/1M1_HF977Hd_-5P-vUWQebg?pwd=j3ek#list/path=%2F&vmode=list>
  — extraction code `j3ek`
- Official source: RGBDT500 / RGBDT dataset by Zhu et al., NeurIPS 2026
  ("Collaborating vision, depth, and thermal signals for multi-modal
  tracking: Dataset and algorithm") — see <https://github.com/RGBDT>
  for the canonical release.
- Use the **official train/test split**; convert annotations to the YOLO
  layout above with `scripts/convert_rgbdt500.py` (depth already aligned).

## 3. DroneVehicle (public, 26,970 RGB–IR pairs, 5 vehicle classes)

- Official: <https://github.com/zhaoxiangyu97/DroneVehicle>
  (Sun et al., IEEE TIP 2022).
- Depth is **synthesized from the stereo pair with Semi-Global Matching**
  (paper Sec. 4.5.5): `scripts/make_sgm_depth.py` (OpenCV `SGBM`) —
  run it once on the extracted RGB pairs, output into `depth/`.
- Classes: `car, bus, truck, van, others`.

## Synthetic sample (no download needed)

```bash
python scripts/make_fake_data.py --out data/fake --n 24
```

Produces the same layout with random geometric scenes (~28.3% invalid depth,
15.5% pseudo-depth) purely for smoke-testing the pipeline — never use it for
scientific numbers.

## Validation quick check

```bash
python scripts/check_dataset.py --root data/urbantraffic3m
```

Reports: triplet alignment, invalid-depth fraction (expect ≈28.3%),
pseudo-depth fraction (expect ≈15.5%), per-batch counts against Table 4
(numeric 1052/620, seq6_3_8 436/193, shuming 370/146, hehe 89/25,
seq_suppl 53/16), and the small-target ratios of the batch table.
