# results/

Outputs of `scripts/visualize.py` (`figures/`) and `scripts/download_data.sh
--verify` (`*.sha256`). Generated locally; only `.gitkeep` and committed
checksum manifests live in git.

Typical contents after a full reproduction run:

```
figures/viz_<stem>.png      Grad-CAM + VADE-B gate + RU-Gate mode masks (Sec. 4.6)
urbantraffic3m.sha256       dataset integrity (paper §4.1.1)
rgbdt500.sha256             "
dronevehicle.sha256         "
```
