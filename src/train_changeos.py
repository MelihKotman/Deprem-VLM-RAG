"""Adım 8 / kutu 3 — ChangeOS (torchange) eğitim + değerlendirme betiği (Kaggle T4, fp16 = autocast + GradScaler).

Örnek (Kaggle, pilot; ayarlama katlaması = Malatya dışarıda, test ve Osmaniye'ye DOKUNMAZ):
    python -m src.train_changeos --scenario S1 --holdout Malatya --pilot --out /kaggle/working/runs/pilot_a
Tam koşu (S1 ana, seed 42; eğitim = Adıyaman+Malatya+Osmaniye; nihai test ayrı komutla):
    python -m src.train_changeos --scenario S1 --epochs 50 --seed 42 --out /kaggle/working/runs/s1_r50_s42
    python -m src.train_changeos --scenario S1 --eval-only --ckpt <.../model_final.pt> --eval-role test --allow-test --out <...>
Tetikleyici ölçümü (docs/adim8_tetikleyici.md; Osmaniye katlaması, seed 42):
    python -m src.train_changeos --scenario S1 --holdout Osmaniye --trigger-fold --epochs 50 --seed 42 --out ...

Ön-kayıt korumaları (src/dm3_dataset.select_tiles): test kilitli (--allow-test), Osmaniye katlaması kilitli (--trigger-fold),
kontrol (Gaziantep) eğitimde yok. Model seçimi YOK: sabit epoch sayısı, son ağırlık (test/katlama üzerinde erken durdurma yapılmaz).
BRIGHT/xBD checkpoint'i YÜKLENMEZ; yalnız ImageNet omurga (ResNet `pretrained=True`).
Çıktılar (--out): config_used.json, log.jsonl (epoch başına), last.pt (devam için), model_final.pt, metrics.json, per_tile.csv, timing.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load
from src.dm3_dataset import (EvalTiles, TrainTiles, collate_eval, collate_train, load_split, select_tiles)
from src.dm3_eval import Acc, object_vote, trigger_r, zone
from src.dm3_verify import find_root


# ---------------------------------------------------------------- model
def build_model(encoder, pretrained):
    from src.changeos_check import build
    return build(encoder, num_dam_classes=4, pretrained=pretrained)


def logits_fn(model, x):
    """ChangeOS.forward'un eğitim yolunun kayıpsız hali (kayıbı kendimiz hesaplıyoruz: ignore + sınıf ağırlığı)."""
    import torchange as tc
    f = tc.bitemporal_forward(model.encoder, x)
    t1, st = model.decoder(*f)
    return model.head.loc_cls(t1), model.head.dam_cls(st)


def loss_fn(loc_logit, dam_logit, dam_t, cls_w):
    import ever.module.loss as L
    loc_logit, dam_logit = loc_logit.float(), dam_logit.float()
    dam = dam_t.long()
    loc_t = (dam_t > 0).float()
    loc_t[dam_t == 255] = 255.0                    # geçersiz piksel: konum hedefi de ignore
    l_bce = L.binary_cross_entropy_with_logits(loc_logit, loc_t)
    l_tv = L.tversky_loss_with_logits(loc_logit, loc_t, alpha=0.9, beta=0.1, gamma=1.0)
    l_ce = F.cross_entropy(dam_logit, dam, weight=cls_w, ignore_index=255)
    l_dc = L.dice_loss_with_logits(dam_logit, dam, ignore_index=255)
    return dict(t1_bce=l_bce, t1_tver=l_tv, c_ce=l_ce, c_dice=l_dc)


# ---------------------------------------------------------------- değerlendirme
@torch.no_grad()
def evaluate(model, loader, dev, fp16, max_tiles=None, save_dir=None):
    model.eval()
    acc = Acc()
    t0 = time.time(); n = 0
    for x, gt, valid, meta in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and dev == "cuda"):
            ll, dl = logits_fn(model, x)
        loc = ll.float().sigmoid().squeeze(1).cpu().numpy()
        dam = dl.float().softmax(1).cpu().numpy()
        for i, m in enumerate(meta):
            pred = object_vote(loc[i], dam[i])
            g = gt[i].numpy(); v = valid[i].numpy()
            keys = [("ALL",), ("group", m["group"]), ("tier", m["tier"]), ("group_tier", m["group"], m["tier"])]
            acc.update(keys, g, pred, v, m)
            n += 1
        if max_tiles and n >= max_tiles:
            break
    s = acc.summary()
    return acc, s, time.time() - t0


def report(s):
    a = s.get("ALL", None)
    if not a:
        return
    for lvl in ("strict", "tol5", "tol15"):
        r = a[lvl]
        print(f"  {lvl:6s} F1 sağlam {r['f1_intact']:.3f} hasarlı {r['f1_damaged']:.3f} yıkık {r['f1_destroyed']:.3f} "
              f"| macro {r['macro_f1']:.3f} | hasar-F1 {r['damage_f1']:.3f} | bileşen {r['n_gt_components']}")
    p = a["pixel"]
    print(f"  piksel IoU sağlam {p['iou_intact']:.3f} hasarlı {p['iou_damaged']:.3f} yıkık {p['iou_destroyed']:.3f} | konum IoU {p['loc_iou']:.3f}")


# ---------------------------------------------------------------- ana
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["S1", "S2"], default="S1")
    ap.add_argument("--holdout", default=None, help="eğitim şehrini dışarıda bırakıp onu değerlendir (örn. Malatya)")
    ap.add_argument("--trigger-fold", action="store_true", help="Osmaniye katlamasını (tetikleyici ölçümü) aç")
    ap.add_argument("--eval-role", choices=["test", "control"], default=None)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--encoder", default="r50", choices=["r50", "r18", "swint"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-tile", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="kırpım sayısı (karo sayısı = batch / crops-per-tile)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--post-shift", type=int, default=0, help="sonrası kırpımı için ±px rastgele öteleme (pilotta seçilir)")
    ap.add_argument("--dam-p", type=float, default=0.5, help="kırpımın hasarlı/yıkık piksel çevresinden seçilme oranı")
    ap.add_argument("--cls-weights", default=None, help="CE sınıf ağırlıkları 'arka,sağlam,hasarlı,yıkık' örn. 1,1,3,3")
    ap.add_argument("--no-photo", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default=None, help="karo önbelleği (Kaggle: /kaggle/working/cache)")
    ap.add_argument("--max-minutes", type=float, default=None, help="epoch sonunda süre aşıldıysa kaydedip dur")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="kısa pilot: az karo, az epoch, süre/bellek ölçümü + kota tahmini")
    ap.add_argument("--pilot-tiles", type=int, default=64)
    ap.add_argument("--eval-max-tiles", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)

    cfg = load()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = dev == "cuda"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True

    root = find_root(a.root)
    if root is None or not Path(root).exists():
        raise SystemExit("veri klasörü bulunamadı (--root veya RAG_DATA)")
    df = load_split(a.scenario, cfg)
    tr_df, ev_df = select_tiles(df, holdout=a.holdout, trigger_fold=a.trigger_fold,
                                eval_role=a.eval_role, allow_test=a.allow_test)
    full_train_n = len(tr_df)
    sanity = False
    if a.pilot and len(tr_df) > a.pilot_tiles:
        tr_df = tr_df.sample(n=a.pilot_tiles, random_state=a.seed).reset_index(drop=True)
    if ev_df is None:                                   # değerlendirme kümesi verilmedi → eğitim içi SANITY (genelleme değil)
        ev_df = tr_df.sample(n=min(24, len(tr_df)), random_state=a.seed).reset_index(drop=True)
        sanity = True
    if a.pilot and len(ev_df) > 40:
        ev_df = ev_df.sample(n=40, random_state=a.seed).reset_index(drop=True)
    if a.eval_max_tiles and len(ev_df) > a.eval_max_tiles:
        ev_df = ev_df.sample(n=a.eval_max_tiles, random_state=a.seed).reset_index(drop=True)

    info = dict(args=vars(a), config_sha1=cfg["_config_sha1"], split_file=cfg["split"][a.scenario],
                split_sha1_config=cfg["split"].get(f"{a.scenario}_sha1"), n_train=len(tr_df), n_train_full=full_train_n,
                n_eval=len(ev_df), eval_is_sanity_in_sample=sanity,
                train_groups=sorted(set(tr_df.group)), eval_groups=sorted(set(ev_df.group)),
                python=platform.python_version(), torch=torch.__version__, device=dev,
                gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, data_root=str(root))
    (out / "config_used.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    print(f"[{a.scenario}] eğitim {len(tr_df)} karo {info['train_groups']} | değerlendirme {len(ev_df)} karo {info['eval_groups']}"
          f"{'  (SANITY: eğitim içi, genelleme değil)' if sanity else ''}")

    model = build_model(a.encoder, pretrained=not a.no_pretrained).to(dev)
    ev_ds = EvalTiles(ev_df, root, cache_dir=a.cache_dir)
    ev_dl = DataLoader(ev_ds, batch_size=1, shuffle=False, num_workers=min(a.workers, 2), collate_fn=collate_eval)

    if a.eval_only:
        sd = torch.load(a.ckpt, map_location=dev)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        acc, s, sec = evaluate(model, ev_dl, dev, fp16)
        _save_metrics(out, acc, s, sec, a, info)
        return 0

    tr_ds = TrainTiles(tr_df, root, crop=a.crop, crops_per_tile=a.crops_per_tile, dam_p=a.dam_p,
                       post_shift_px=a.post_shift, photo=not a.no_photo, cache_dir=a.cache_dir, seed=a.seed)
    tiles_per_batch = max(1, a.batch // a.crops_per_tile)
    g = torch.Generator(); g.manual_seed(a.seed)
    tr_dl = DataLoader(tr_ds, batch_size=tiles_per_batch, shuffle=True, num_workers=a.workers, drop_last=True,
                       collate_fn=collate_train, generator=g, persistent_workers=a.workers > 0, pin_memory=dev == "cuda")
    steps_per_epoch = len(tr_dl)
    total_steps = a.epochs * steps_per_epoch
    cls_w = None
    if a.cls_weights:
        cls_w = torch.tensor([float(v) for v in a.cls_weights.split(",")], device=dev)
        assert len(cls_w) == 4

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = int(a.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    scaler = torch.amp.GradScaler(enabled=fp16)

    ep0 = 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); ep0 = ck["epoch"]
        print(f"devam: epoch {ep0}/{a.epochs}")

    t_start = time.time()
    epoch_times = []
    for ep in range(ep0, a.epochs):
        model.train(); tr_ds.set_epoch(ep)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_ep = time.time(); wait = 0.0; gpu = 0.0; acc_l = {}; nb = 0
        t_prev = time.time()
        for x, dam in tr_dl:
            wait += time.time() - t_prev
            tg = time.time()
            x = x.to(dev, non_blocking=True); dam = dam.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                ll, dl = logits_fn(model, x)
            ld = loss_fn(ll, dl, dam, cls_w)
            loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            gpu += time.time() - tg
            for k, v in ld.items():
                acc_l[k] = acc_l.get(k, 0.0) + float(v)
            nb += 1
            if not np.isfinite(float(loss)):
                raise SystemExit(f"kayıp sonlu değil (epoch {ep}, adım {nb}) — fp16/öğrenme oranı kontrol et")
            t_prev = time.time()
        et = time.time() - t_ep; epoch_times.append(et)
        rec = dict(epoch=ep + 1, sec=round(et, 1), data_wait_s=round(wait, 1), gpu_s=round(gpu, 1), steps=nb,
                   crops_per_s=round(nb * tiles_per_batch * a.crops_per_tile / et, 2), lr=sched.get_last_lr()[0],
                   peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None,
                   **{k: round(v / max(nb, 1), 4) for k, v in acc_l.items()})
        print(json.dumps(rec, ensure_ascii=False))
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep + 1), out / "last.pt")
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes and ep + 1 < a.epochs:
            print(f"--max-minutes aşıldı; epoch {ep + 1}'de duruldu. Devam: aynı komut + --resume")
            return 3
    torch.save(model.state_dict(), out / "model_final.pt")

    acc, s, sec = evaluate(model, ev_dl, dev, fp16)
    mean_ep = float(np.mean(epoch_times)) if epoch_times else None
    last = json.loads((out / "log.jsonl").read_text().strip().splitlines()[-1])
    timing = dict(epochs_run=len(epoch_times), mean_epoch_s=mean_ep, steps_per_epoch=steps_per_epoch,
                  data_wait_frac=round(last["data_wait_s"] / last["sec"], 3), eval_s=round(sec, 1), eval_tiles=len(ev_df),
                  eval_s_per_tile=round(sec / max(len(ev_df), 1), 2), peak_mem_GB=last["peak_mem_GB"],
                  train_tiles=len(tr_df), train_tiles_full=full_train_n)
    if a.pilot and mean_ep:
        scale = full_train_n / max(len(tr_df), 1)
        ep_full = mean_ep * scale
        timing["EXTRAPOLATION_NOT_MEASURED_AT_FULL_SIZE"] = dict(
            epoch_s_full=round(ep_full), hours_per_50_epochs=round(ep_full * 50 / 3600, 2),
            note="pilot epoch süresi × (tam eğitim karosu / pilot karosu); değerlendirme ve kayıt süresi dahil değil")
    (out / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1))
    print("timing:", json.dumps(timing, ensure_ascii=False))
    _save_metrics(out, acc, s, sec, a, info)
    return 0


def _save_metrics(out, acc, s, sec, a, info):
    all_ = s.get("ALL")
    res = dict(eval_seconds=round(sec, 1), eval_is_sanity_in_sample=info["eval_is_sanity_in_sample"],
               eval_groups=info["eval_groups"], by=s)
    if a.trigger_fold and a.holdout and a.holdout.lower().startswith('osmaniye') and all_:
        r = trigger_r(all_)
        res["trigger"] = dict(r=r, zone=zone(r), strict_damage_f1=all_["strict"]["damage_f1"],
                              tol15_damage_f1=all_["tol15"]["damage_f1"], note="ön-kayıtlı; docs/adim8_tetikleyici.md")
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pd.DataFrame(acc.tile_rows).to_csv(out / "per_tile.csv", index=False)
    print("metrikler ->", out / "metrics.json")
    report(s)
    if "trigger" in res:
        print("tetikleyici:", res["trigger"])


if __name__ == "__main__":
    raise SystemExit(main())
