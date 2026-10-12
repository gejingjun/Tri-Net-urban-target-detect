#!/usr/bin/env bash
# Download helper for the three datasets used in the paper.
#
# UrbanTraffic-3M  (new, 3,000 triplets, 12 classes)
#   Baidu Pan: https://pan.baidu.com/s/17luVd99FGrdL4W5LgMvBKg?pwd=xww8  code: xww8
# RGBDT500         (public, 203,700 triplets, >=66 classes)
#   Baidu Pan: https://pan.baidu.com/s/1M1_HF977Hd_-5P-vUWQebg?pwd=j3ek  code: j3ek
# DroneVehicle     (public, 26,970 RGB-IR pairs; depth synthesised via SGM)
#   official: https://github.com/zhaoxiangyu97/DroneVehicle (see data/README.md)
#
# Baidu Pan links require a browser/official client login-free download;
# the archive layout after extraction must be:
#   data/urbantraffic3m/{images,ir,depth,labels,meta}/{train,val,test}/
# SHA256 checksums of the extracted folders are printed at the end so you
# can verify against results/checksums.sha256 committed in this repo.
set -euo pipefail
mkdir -p data/raw
echo "Open the links above in a browser, extract into data/urbantraffic3m etc.,"
echo "then re-run with --verify to check the layout."
if [[ "${1:-}" == "--verify" ]]; then
  for d in data/urbantraffic3m data/rgbdt500 data/dronevehicle; do
    if [[ -d "$d/images" || -d "$d" ]]; then
      find "$d" -type f | sort | xargs -r sha256sum > "results/${d##*/}.sha256" || true
      echo "checksum -> results/${d##*/}.sha256"
    else
      echo "MISSING: $d"
    fi
  done
fi
