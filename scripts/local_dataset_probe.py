# -*- coding: utf-8 -*-
"""Local-dataset stats probe (read-only): dtypes, invalid-depth fraction,
pseudo-depth share, label counts for the on-disk UrbanTraffic-3M copy."""
import os, random, collections
import numpy as np, cv2

ROOT = r"C:\Users\Administrator\Desktop\CITYTARGET\初赛数据集-面向城市场景的多模态目标检测"


def imread_u(p, flag=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, flag)


for split, folder in (("train", "训练集\\train"), ("test", "测试集\\test")):
    base = os.path.join(ROOT, folder)
    print(f"== {split} ==")
    for m in ("visible", "infrared", "depth"):
        files = sorted(os.listdir(os.path.join(base, m)))
        print(f"  {m}: {len(files)}")
    lp = os.path.join(base, "labels")
    labs = sorted(os.listdir(lp)) if os.path.isdir(lp) else []
    print(f"  labels: {len(labs)}")
    random.seed(0)
    dfiles = sorted(os.listdir(os.path.join(base, "depth")))
    sample = random.sample(dfiles, min(200, len(dfiles)))
    zf, mx, dtset, d255 = [], [], set(), 0
    for fn in sample:
        im = imread_u(os.path.join(base, "depth", fn))
        if im is None:
            continue
        if im.ndim == 3:
            im = im[..., 0]
        zf.append(float((im == 0).mean()))
        mx.append(int(im.max()))
        dtset.add(str(im.dtype))
        if im.max() <= 255 and im.dtype == np.uint8:
            d255 += 1
    print(f"  depth dtypes: {dtset}")
    print(f"  invalid-depth fraction (mean of {len(sample)} sampled): {np.mean(zf):.4f}")
    print(f"  uint8 pseudo-depth share in sample: {d255}/{len(sample)}")
    # batch prefixes
    pref = collections.Counter()
    for fn in sorted(os.listdir(os.path.join(base, "visible"))):
        stem = os.path.splitext(fn)[0]
        pref[stem.split("_")[0] if not stem[0].isdigit() else "numeric"] += 1
    print("  prefixes:", dict(pref))
    # label instance counts
    n_inst, n_small = 0, 0
    per_cls = collections.Counter()
    if labs:
        for fn in labs:
            rows = np.loadtxt(os.path.join(lp, fn), dtype=np.float32, ndmin=2)
            if rows.size == 0:
                continue
            if rows.ndim == 1:
                rows = rows[None]
            n_inst += len(rows)
            per_cls.update(rows[:, 0].astype(int).tolist())
            H = W = None
            im = imread_u(os.path.join(base, "visible", fn.replace(".txt", ".jpg")))
            if im is not None:
                H, W = im.shape[:2]
                areas = rows[:, 3] * W * rows[:, 4] * H
                n_small += int((areas < 32 ** 2).sum())
    print(f"  instances: {n_inst}, small(<32^2): {n_small}")
    print(f"  per-class: {dict(sorted(per_cls.items()))}")
