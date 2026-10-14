#!/usr/bin/env python3
"""Link the LOCAL UrbanTraffic-3M copy into the repo layout (no copying).

Detected source (this machine):
    <CITYTARGET>/初赛数据集-面向城市场景的多模态目标检测/
      训练集/train/{visible,infrared,depth,labels}   2000 triplets + labels
      测试集/test/{visible,infrared,depth}            1000 triplets
    <CITYTARGET>/labels/                             1000 test-set label .txt

Target layout (symlinks / junctions + copied text files):
    data/urbantraffic3m/
      images/{train,test} -> visible dirs
      ir/{train,test}     -> infrared dirs
      depth/{train,test}  -> depth dirs
      labels/{train,test} -> label dirs
      meta/{train,test}.json  {"<stem>": {"batch": ..., "condition": ...}}

Usage: python scripts/link_local_dataset.py [--src <path>]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CANDIDATES = [
    r"C:\Users\Administrator\Desktop\CITYTARGET\初赛数据集-面向城市场景的多模态目标检测",
]
BATCH_PAT = re.compile(r"^(shuming|hehe|seq6_3_8|seq_suppl|numeric)")


def batch_of(stem: str) -> str:
    m = BATCH_PAT.match(stem)
    if m:
        return m.group(1)
    return "numeric" if stem[0].isdigit() else stem.split("_")[0]


def condition_of(stem: str, meta_hint: str | None = None) -> str:
    # condition labels are not shipped per-file; leave unknown -> scene
    # classifier self-supervised path (paper Sec. 3.5 permits missing labels)
    return "unknown"


def link_dir(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if os.name == "nt":
        # junction works without admin privileges on Windows
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           capture_output=True, text=True)
        return "ok" if dst.exists() else f"fail: {r.stderr.strip()[:120]}"
    dst.symlink_to(src, target_is_directory=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default="data/urbantraffic3m")
    a = ap.parse_args()
    src = Path(a.src or next((c for c in CANDIDATES if Path(c).exists()), ""))
    if not src.exists():
        sys.exit("local dataset not found; pass --src")
    root = Path(a.dst)
    mapping = {
        "train": {
            "images": src / "训练集/train/visible",
            "ir": src / "训练集/train/infrared",
            "depth": src / "训练集/train/depth",
            "labels": src / "训练集/train/labels",
        },
        "test": {
            "images": src / "测试集/test/visible",
            "ir": src / "测试集/test/infrared",
            "depth": src / "测试集/test/depth",
            "labels": Path(r"C:\Users\Administrator\Desktop\CITYTARGET/labels"),
        },
    }
    for split, mods in mapping.items():
        for mod, s in mods.items():
            if not s.exists():
                print(f"  [warn] missing {s}")
                continue
            print(f"  {mod}/{split} <- {s}:", link_dir(s, root / mod / split))
    # meta json (batch tags; condition unknown)
    (root / "meta").mkdir(exist_ok=True)
    for split in ("train", "test"):
        ids = sorted(p.stem for p in (root / "images" / split).iterdir())
        out = {st: {"batch": batch_of(st), "condition": condition_of(st)}
               for st in ids}
        (root / "meta" / f"{split}.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        cnt = Counter(v["batch"] for v in out.values())
        print(f"  meta/{split}.json: {len(out)} entries, batches={dict(cnt)}")
    print(f"linked -> {root.resolve()}")


if __name__ == "__main__":
    main()
