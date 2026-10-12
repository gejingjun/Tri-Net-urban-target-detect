"""Split protocol (paper Sec. 4.1.1 UrbanTraffic-3M):

  '2,000 / 1,000 train/validation, batch-stratified and sequence-grouped
   to prevent information leakage'

Five acquisition batches: numeric, seq6_3_8, shuming, hehe, seq_suppl.
Sequences: file stems like 'seq6_3_8_00000123' group frames of one capture
sequence; all frames of one sequence land on the SAME side of the split.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def sequence_of(stem: str) -> str:
    """Group key: leading '<batch>_<seqidx>' prefix before the frame number."""
    m = re.match(r"^(.*?)(\d{4,})$", stem)
    if not m:
        return stem
    prefix = m.group(1).rstrip("_")
    # e.g. 'shuming_82_00000181' -> seq group 'shuming_82'
    parts = prefix.split("_")
    return "_".join(parts[:-1]) + "_" + parts[-1] if len(parts) >= 2 else prefix


def batch_of(stem: str) -> str:
    for b in ("numeric", "seq6_3_8", "shuming", "hehe", "seq_suppl"):
        if stem.startswith(b):
            return b
    return stem.split("_")[0]


def batch_stratified_split(root: Path, split_dir: Path, train_ratio=2 / 3,
                           seed=2026) -> dict:
    """Write meta/{train,val,test}.json with per-stem batch + sequence tags,
    honouring the paper's batch-stratified, sequence-grouped protocol.

    Idempotent: if meta files already exist (official protocol shipped with
    the dataset), they are used as-is.
    """
    meta_dir = split_dir / "meta"
    existing = {k: meta_dir / f"{k}.json" for k in ("train", "val", "test")}
    if all(p.exists() for p in existing.values()):
        return {k: json.loads(p.read_text(encoding="utf-8")) for k, p in existing.items()}
    img_dir = split_dir / "images"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    # group stems by sequence, stratify by batch ratio
    import random
    rng = random.Random(seed)
    by_batch = defaultdict(list)
    for s in stems:
        by_batch[batch_of(s)].append(s)
    assign = {}
    for b, items in by_batch.items():
        seqs = defaultdict(list)
        for it in items:
            seqs[sequence_of(it)].append(it)
        keys = sorted(seqs)
        rng.shuffle(keys)
        n_train = round(len(items) * train_ratio)
        seen, train_pool = 0, []
        for k in keys:
            if seen < n_train:
                train_pool.extend(seqs[k])
                seen += len(seqs[k])
                for x in seqs[k]:
                    assign[x] = "train"
            else:
                for x in seqs[k]:
                    assign[x] = "val"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = {"train": {}, "val": {}, "test": {}}
    for s, side in assign.items():
        out[side][s] = {"batch": batch_of(s), "sequence": sequence_of(s)}
    # empty test.json (external eval batches are documented per release)
    for k, p in existing.items():
        p.write_text(json.dumps(out[k], ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return out
