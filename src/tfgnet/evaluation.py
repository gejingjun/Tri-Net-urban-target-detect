"""COCO-protocol evaluation: mAP@50, mAP@50:95, per-class AP, AP_S.

Pure-numpy 101-point interpolation (COCO style); no pycocotools required at
runtime, but predictions can also be dumped in COCO JSON for cross-checking
against the official pycocotools evaluator:

    python scripts/eval.py --dump-coco-pred runs/x/preds.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

IOUS = np.arange(0.5, 1.0, 0.05)


def _iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    ix = np.maximum(A[:, None, 0], B[None, :, 0])
    iy = np.maximum(A[:, None, 1], B[None, :, 1])
    xx = np.minimum(A[:, None, 2], B[None, :, 2])
    xy = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(xx - ix, 0, None) * np.clip(xy - iy, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def _area(xyxy):
    return (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])


def _rec_prec(scores, matches, n_gt, area_mask=None):
    if len(scores) == 0:
        return np.zeros(101)
    order = np.argsort(-scores)
    tp = np.zeros(len(scores) + 1)
    fp = np.zeros(len(scores) + 1)
    for i, idx in enumerate(order, 1):
        if matches[idx]:
            tp[i] = tp[i - 1] + 1
        else:
            fp[i] = fp[i - 1] + 1
    tp, fp = tp[1:], fp[1:]
    prec = tp / (tp + fp + 1e-9)
    rec = tp / max(n_gt, 1)
    # 101-pt interpolation
    q = np.linspace(0, 1, 101)
    ap = np.interp(q, rec, prec, left=0, right=0)
    return ap


def evaluate(preds, gts, ious=IOUS, small_area=32 ** 2):
    """preds: list of dict(image_id, boxes[N,4] xyxy, scores[N], cls[N])
       gts:   list of dict(image_id, boxes, cls, areas=None)
    Returns dict(mAP50, mAP50_95, per_class {c: ap50}, AP_S)."""
    gt_by_img = {g["image_id"]: g for g in gts}
    classes = sorted(set(np.concatenate([g["cls"] for g in gts] or [[]]).tolist())
                     if gts else set())
    stats = defaultdict(list)
    small_tp = defaultdict(lambda: [0, 0])
    for cid in classes:
        det, n_gt = [], 0
        for img_id, g in gt_by_img.items():
            gm = g["cls"] == cid
            gt_boxes = g["boxes"][gm]
            n_gt += int(gm.sum())
            p = next((x for x in preds if x["image_id"] == img_id), None)
            pm = (p["cls"] == cid) if p else np.zeros(0, bool)
            pb = p["boxes"][pm] if p else np.zeros((0, 4))
            ps = p["scores"][pm] if p else np.zeros(0)
            det.append((img_id, gt_boxes, pb, ps))
        # IoU matching per iou threshold
        aps = {}
        for thr in ious:
            all_sc, all_m = [], []
            for img_id, gb, pb, ps in det:
                if len(gb):
                    iou = _iou_matrix(pb, gb)
                    matched = set()
                    m = np.zeros(len(pb), bool)
                    for j in np.argsort(-ps):
                        if len(iou) == 0:
                            continue
                        k = iou[j].argmax()
                        if iou[j, k] >= thr and k not in matched:
                            m[j] = True
                            matched.add(int(k))
                else:
                    m = np.zeros(len(pb), bool)
                if len(ps):
                    all_sc.append(ps)
                    all_m.append(m)
            sc = np.concatenate(all_sc) if all_sc else np.zeros(0)
            mm = np.concatenate(all_m) if all_m else np.zeros(0, bool)
            aps[thr] = _rec_prec(sc, mm, n_gt)
        stats[cid] = aps
    iou50 = np.array([stats[c][0.5].mean() for c in classes]) if classes else np.zeros(0)
    iouall = np.array([np.mean([v.mean() for v in stats[c].values()]) for c in classes]) if classes else np.zeros(0)
    # small-target mAP@50 (area < 32^2)
    return dict(
        mAP50=float(iou50.mean() * 100) if classes else 0.0,
        mAP50_95=float(iouall.mean() * 100) if classes else 0.0,
        per_class={int(c): float(v[0.5].mean() * 100) for c, v in stats.items()},
        AP_S=None,  # computed by evaluate_with_areas() when boxes provided
        n_gt_total=int(sum(int((g['cls'] >= 0).sum()) for g in gts)),
    )


def evaluate_with_areas(preds, gts, small_area=32 ** 2):
    """mAP + AP_S (COCO small: area < 32^2)."""
    res = evaluate(preds, gts)
    # AP_S: restrict both GT and predictions to small boxes
    gts_s = []
    for g in gts:
        a = _area(g["boxes"]) if len(g["boxes"]) else np.zeros(0)
        m = a < small_area
        gts_s.append(dict(image_id=g["image_id"], boxes=g["boxes"][m], cls=g["cls"][m]))
    preds_s = []
    for p in preds:
        a = _area(p["boxes"]) if len(p["boxes"]) else np.zeros(0)
        m = a < small_area
        preds_s.append(dict(image_id=p["image_id"], boxes=p["boxes"][m],
                            scores=p["scores"][m], cls=p["cls"][m]))
    res_small = evaluate(preds_s, [g for g in gts_s if len(g["boxes"])], ious=np.array([0.5]))
    res["AP_S"] = res_small["mAP50"]
    return res


def dump_coco_json(preds, out_path, class_names=None):
    anns, imgs = [], []
    aid = 0
    for i, p in enumerate(preds):
        imgs.append({"id": p["image_id"], "file_name": f"{p['image_id']}.png"})
        for b, s, c in zip(p["boxes"], p["scores"], p["cls"]):
            aid += 1
            anns.append({"id": aid, "image_id": p["image_id"], "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]),
                                  float(b[2] - b[0]), float(b[3] - b[1])],
                         "area": float((b[2] - b[0]) * (b[3] - b[1])),
                         "score": float(s), "iscrowd": 0})
    cats = [{"id": j + 1, "name": (class_names or {})[j] if class_names else str(j)}
            for j in range(max([a["category_id"] for a in anns], default=1))]
    json.dump({"images": imgs, "annotations": anns, "categories": cats},
              open(out_path, "w", encoding="utf-8"))
    return out_path
