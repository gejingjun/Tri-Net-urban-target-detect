"""Trainer: SGD + cosine schedule + warmup (paper Sec. 3.6).

    SGD, lr0=0.01, momentum=0.937, wd=5e-4, cosine annealing,
    300 epochs, 10-epoch warmup, batch 16, 640x640,
    Mosaic + MixUp + hflip + HSV, depth modality dropout p=0.15,
    mean±std over seeds {2026, 3407, 7102}.

Supports:
  * TFGNet tri-modal training (Eq. 32 total loss)
  * SingleBranchNet baselines (YOLOv8s single/early-fusion)
  * condition/batch filters and degradation modes (robustness tables)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.augment import TriModalAugmentor, mixup_pair
from .data.dataset import TriModalDataset, collate_tri
from .evaluation import evaluate_with_areas
from .losses import AuxiliaryLosses, DetectionLoss
from .model import TFGNet, build_model
from .utils import get_logger, set_seed


class Trainer:
    def __init__(self, cfg: Config, run_name="run", resume=None):
        self.cfg = cfg
        self.log = get_logger("tfgnet.train")
        set_seed(int(cfg.train.seeds[0]) if cfg.train.seeds else 2026)
        self.device = torch.device(
            "cuda" if (cfg.train.device == "auto" and torch.cuda.is_available())
            else cfg.train.device if cfg.train.device != "auto" else "cpu")
        # encoder shared by train/val (fit on train)
        mode = getattr(cfg.model, "depth_encoding_mode", "vade_a")
        from .modules.vade import DepthEncoder
        fixed_edges = cfg.model.fixed_edges_mm if mode == "near_mid_far" else None
        self.encoder = DepthEncoder(cfg.model.c_d, cfg.model.d_min_mm,
                                    cfg.model.d_max_mm, cfg.model.quantiles,
                                    mode=mode, fixed_edges=fixed_edges)
        aug = TriModalAugmentor(img_size=cfg.data.img_size,
                                **{k: v for k, v in cfg.data.augment.items()
                                   if k in ("mosaic", "mixup", "hflip", "hsv_h", "hsv_s", "hsv_v")})
        self.train_ds = TriModalDataset(cfg.data.root, cfg.data.train_split,
                                        cfg.data.img_size, self.encoder, aug,
                                        nc=cfg.data.nc)
        self.val_ds = TriModalDataset(cfg.data.root, cfg.data.val_split,
                                      cfg.data.img_size, self.encoder, None,
                                      nc=cfg.data.nc)
        self.train_ds.bind_mosaic_pool()
        self.train_ds.fit_encoder()
        self.model = build_model(cfg).to(self.device)
        self.det_loss = DetectionLoss(nc=cfg.data.nc).to(self.device)
        self.aux_loss = AuxiliaryLosses(cfg.train.lambda_valid,
                                        cfg.train.lambda_scene).to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg.train.lr0,
                                   momentum=cfg.train.momentum,
                                   weight_decay=cfg.train.weight_decay)
        self.epochs = cfg.train.epochs
        self.warmup = cfg.train.warmup_epochs
        self.scaler = torch.amp.GradScaler("cuda") if cfg.train.amp and self.device.type == "cuda" else None
        self.run_dir = Path(cfg.train.save_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self.load(resume)

    # ------------------------------------------------------------------
    def loader(self):
        g = torch.Generator(); g.manual_seed(0)
        return DataLoader(self.train_ds, batch_size=self.cfg.train.batch,
                          shuffle=True, num_workers=self.cfg.train.workers,
                          collate_fn=collate_tri, drop_last=True,
                          generator=g, pin_memory=(self.device.type == "cuda"))

    def lr_at(self, ep, it, n_it):
        t = ep + it / max(n_it, 1)
        warm = t / max(self.warmup, 1)
        cos = 0.5 * (1 + math.cos(math.pi * max(0.0, (t - self.warmup)) /
                                  max(self.epochs - self.warmup, 1)))
        return self.cfg.train.lr0 * (warm if t < self.warmup else cos)

    # ------------------------------------------------------------------
    def train_step(self, batch):
        m = self.model
        rgb = batch["rgb"].to(self.device, non_blocking=True)
        ir = batch["ir"].to(self.device, non_blocking=True)
        e = batch["e_dep"].to(self.device, non_blocking=True)
        targets = {k: batch[k] for k in ("cls", "boxes", "n")}
        outs, aux = m(rgb, ir, e)
        losses = self.det_loss(outs, targets)
        total = losses["loss_cls"] + losses["loss_box"] + losses["loss_dfl"]
        # L_valid
        if isinstance(m, TFGNet) and aux.get("g_pixel") is not None:
            total = total + self.aux_loss.validity(aux["g_pixel"],
                                                   batch["valid"].to(self.device))
            if aux.get("scene") is not None and batch.get("condition") is not None:
                scene_ids = torch.tensor(
                    [_cond_to_idx(c) for c in batch["condition"]],
                    dtype=torch.long, device=self.device)
                total = total + self.aux_loss.scene(aux["scene"], scene_ids)
        return total, {k: float(v) for k, v in losses.items()}

    @torch.no_grad()
    def predict_batch(self, batch, conf=0.001):
        m = self.model
        rgb = batch["rgb"].to(self.device); ir = batch["ir"].to(self.device)
        e = batch["e_dep"].to(self.device)
        if isinstance(m, TFGNet):
            det, _ = m(rgb, ir, e)
            per = m.head.decode(det, conf_thres=conf)
        else:
            det, _ = m(rgb)
            per = m.head.decode(det, conf_thres=conf)
        preds = []
        from .modules.head import xyxy_nms
        for bi, (b, s, c) in enumerate(per):
            k = xyxy_nms(b, s, c, iou_thres=0.65)
            stem = batch["stem"][bi]
            preds.append(dict(image_id=stem, boxes=b[k].cpu().numpy(),
                              scores=s[k].cpu().numpy(), cls=c[k].cpu().numpy()))
        return preds

    def evaluate(self, conf=0.001):
        self.model.eval()
        dl = DataLoader(self.val_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_tri)
        preds, gts = [], []
        for batch in dl:
            preds += self.predict_batch(batch, conf)
            n = int(batch["n"][0])
            boxes = batch["boxes"][0, :n].numpy() * self.cfg.data.img_size
            xyxy = torch.from_numpy(boxes)
            xyxy = torch.cat([xyxy[:, :2] - xyxy[:, 2:] / 2,
                              xyxy[:, :2] + xyxy[:, 2:] / 2], 1)
            gts.append(dict(image_id=batch["stem"][0], boxes=xyxy.numpy(),
                            cls=batch["cls"][0, :n].numpy()))
        res = evaluate_with_areas(preds, gts)
        self.model.train()
        return res

    # ------------------------------------------------------------------
    def fit(self):
        self.model.train()
        dl = self.loader()
        best = -1.0
        for ep in range(self.epochs):
            t0 = time.time()
            tot = 0.0
            for it, batch in enumerate(dl):
                lr = self.lr_at(ep, it, len(dl))
                for gp in self.opt.param_groups:
                    gp["lr"] = lr
                self.opt.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    with torch.autocast("cuda"):
                        loss, parts = self.train_step(batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.train.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss, parts = self.train_step(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.train.grad_clip)
                    self.opt.step()
                tot += float(loss.detach())
            msg = f"epoch {ep+1}/{self.epochs} loss {tot/max(len(dl),1):.4f} lr {lr:.5f} {time.time()-t0:.1f}s"
            self.log.info(msg)
            if (ep + 1) % 10 == 0 or ep == self.epochs - 1:
                res = self.evaluate()
                self.log.info("val mAP50 %.2f mAP50-95 %.2f AP_S %s",
                              res["mAP50"], res["mAP50_95"], res["AP_S"])
                if res["mAP50"] > best:
                    best = res["mAP50"]
                    self.save(self.run_dir / "best.pt", res)
                self.save(self.run_dir / "last.pt", res)
        return best

    def save(self, path, res=None):
        torch.save({"model": self.model.state_dict(),
                    "encoder": self.encoder.state_dict(),
                    "cfg": self.cfg.to_dict(), "res": res}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"], strict=False)
        if "encoder" in ck:
            self.encoder.load_state_dict(ck["encoder"])
        return ck


def _cond_to_idx(c: str) -> int:
    from .modules.rugate import SCENE_TYPES
    c = (c or "unknown").lower()
    key = "daytime-clear" if ("clear" in c and "night" not in c) else \
          "nighttime" if "night" in c else \
          "fog" if ("fog" in c or "haze" in c) else \
          "glare" if "glare" in c else \
          "backlight" if "back" in c else -1
    return SCENE_TYPES.index(key) if key in SCENE_TYPES else -1
