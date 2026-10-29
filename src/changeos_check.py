"""Adım 8 / kutu 3 — ChangeOS (torchange) kurulabilirlik + T4/fp16 duman testi.

Amaç: ChangeOS'un Kaggle'da kurulup çalıştığını ve süre/bellek ölçümünü almak.
Gerçek veri KULLANMAZ (rastgele tensör). Sonuç: JSON satırı (stdout) + dosya.

Kullanım (Kaggle, GPU T4, internet açık):
    pip install -q ever-beta timm "git+https://github.com/Z-Zheng/pytorch-change-models.git"
    python -m src.changeos_check --encoder r50 --crop 512 --batch 4 --steps 20 --fp16

Sınıf sayısı: damage head 4 (arka plan + sağlam/hasarlı/yıkık); xBD varsayılanı 5'tir.
BRIGHT/xBD checkpoint'i YÜKLENMEZ; yalnız ImageNet omurga ağırlığı (pretrained=True).
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch


def build(encoder: str, num_dam_classes: int, pretrained: bool):
    import ever as er
    from torchange.models import changeos as _reg  # noqa: F401  (modeli er.registry'ye kaydeder)

    if encoder == "r50":
        from torchange.configs.changeos import cos_r50 as c
    elif encoder == "r18":
        from torchange.configs.changeos import cos_r18 as c
    elif encoder == "swint":
        from torchange.configs.changeos import cos_swint as c
    else:
        raise SystemExit(f"bilinmeyen encoder: {encoder}")
    cfg = copy.deepcopy(c.config["model"])
    p = cfg["params"]
    if "head" in p:
        p["head"]["dam_head"]["num_classes"] = num_dam_classes
    if encoder != "swint" and not pretrained:
        p["encoder"]["params"]["pretrained"] = False
    if encoder == "swint" and not pretrained:
        p["encoder"]["params"]["weights"] = None
    return er.builder.make_model(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="r50", choices=["r50", "r18", "swint"])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--classes", type=int, default=4, help="damage head: arka plan + 3 sınıf")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = dict(
        python=platform.python_version(), torch=torch.__version__, device=dev,
        gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
        encoder=a.encoder, crop=a.crop, batch=a.batch, fp16=bool(a.fp16 and dev == "cuda"),
        damage_classes=a.classes,
    )
    import torchange
    res["torchange"] = getattr(torchange, "__version__", "?")

    t0 = time.time()
    model = build(a.encoder, a.classes, pretrained=not a.no_pretrained).to(dev)
    res["build_s"] = round(time.time() - t0, 1)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=res["fp16"])
    B, S = a.batch, a.crop

    def batch():
        x = torch.randn(B, 6, S, S, device=dev)
        loc = (torch.rand(B, S, S, device=dev) > 0.9).long()
        dam = torch.randint(0, a.classes, (B, S, S), device=dev) * loc
        return x, dict(masks=[loc, dam])

    model.train()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for i in range(a.steps):
        x, y = batch()
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=res["fp16"]):
            ld = model(x, y)
        loss = sum(ld.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= 2:  # ısınma adımlarını at
            times.append(time.time() - t)
        losses.append(float(loss))
    res["loss_keys"] = list(ld.keys())
    res["loss_first"], res["loss_last"] = round(losses[0], 4), round(losses[-1], 4)
    res["loss_finite"] = bool(np.isfinite(losses).all())
    res["s_per_step"] = round(float(np.mean(times)), 3) if times else None
    res["img_per_s"] = round(B / float(np.mean(times)), 2) if times else None
    if dev == "cuda":
        res["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # çıkarım: piksel ve nesne tabanlı
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=res["fp16"]):
        x, _ = batch()
        out = model(x)
    res["infer_out_type"] = type(out).__name__
    res["has_object_based_infer"] = hasattr(model, "head") and hasattr(model.head, "object_based_infer") or hasattr(model, "object_based_infer")
    res["status"] = "OK"
    line = json.dumps(res, ensure_ascii=False)
    print(line)
    if a.out:
        with open(a.out, "w") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
