"""Sentetik DisasterM3 benzeri karolar (yalnız hat/boru testi için; gerçek veri DEĞİL).

    python -m src.synth_tiles --out /tmp/fake --per-group 4
    RAG_ROOT=/tmp/fake/repo RAG_DATA=/tmp/fake/data python -m src.train_changeos --scenario S1 --holdout Malatya --pilot ...

Gerçek bölme CSV'sinden her gruptan `per-group` karo alır; aynı dosya adı düzeniyle PNG'ler yazar (öncesi, sonrası, 3 maske).
Dikdörtgen "binalar": sağlam / hasarlı / yıkık; sonrası görüntü `kayma` px kaydırılmış; bazı karolarda öncesi boş şerit (geçerlilik testi).
Bilinen cevap: sonrası, yıkık binalarda parlak/gürültülü → basit bir model bunu öğrenebilmeli (kayıp düşmeli).
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import load, root as repo_root


def make_tile(rng, S=1024, shift=6, blank_strip=False):
    pre = (rng.normal(110, 18, (S, S, 3))).clip(0, 255).astype(np.uint8)
    pre = cv2.GaussianBlur(pre, (0, 0), 3)
    post = pre.copy()
    masks = {c: np.zeros((S, S), np.uint8) for c in ("intact", "damaged", "destroyed")}
    n = int(rng.integers(25, 60))
    for _ in range(n):
        w, h = int(rng.integers(20, 60)), int(rng.integers(20, 60))
        x, y = int(rng.integers(0, S - w)), int(rng.integers(0, S - h))
        cls = rng.choice(["intact", "damaged", "destroyed"], p=[0.7, 0.15, 0.15])
        col = rng.integers(150, 230, 3)
        cv2.rectangle(pre, (x, y), (x + w, y + h), tuple(int(c) for c in col), -1)
        cv2.rectangle(post, (x, y), (x + w, y + h), tuple(int(c) for c in col), -1)
        if cls == "damaged":
            post[y:y + h // 2, x:x + w] = (post[y:y + h // 2, x:x + w] * 0.5).astype(np.uint8)
        elif cls == "destroyed":
            post[y:y + h, x:x + w] = rng.integers(40, 255, (h, w, 3)).astype(np.uint8)
        masks[cls][y:y + h, x:x + w] = 255
    M = np.float32([[1, 0, shift], [0, 1, -shift]])
    post = cv2.warpAffine(post, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    if blank_strip:
        pre[:, : S // 8] = 0
    return pre, post, masks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load()
    R = repo_root()
    out = Path(a.out)
    data, repo = out / "data", out / "repo"
    (repo / "config").mkdir(parents=True, exist_ok=True)
    (repo / "data_manifest" / "splits").mkdir(parents=True, exist_ok=True)
    shutil.copy(R / "config" / "base.yaml", repo / "config" / "base.yaml")
    rng = np.random.default_rng(a.seed)
    val = pd.read_csv(R / cfg["split"]["validity"])
    keep_all = []
    for sc in ("S1", "S2"):
        df = pd.read_csv(R / cfg["split"][sc])
        sub = pd.concat([g.sample(n=min(a.per_group, len(g)), random_state=a.seed) for _, g in df.groupby("group")])
        sub.to_csv(repo / cfg["split"][sc], index=False)
        keep_all.append(sub[["event", "tile", "split"]])
    keep = pd.concat(keep_all).drop_duplicates()
    for r in keep.itertuples():
        blank = rng.random() < 0.2
        pre, post, masks = make_tile(rng, a.size, shift=int(rng.integers(0, 12)), blank_strip=blank)
        b, idr, pf = (data / "train", "train_images", "train") if r.split == "train" else (data / "bench", "test_images", "test")
        suf = r.tile.split("_", 2)[2]
        (b / idr).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(b / idr / f"turkey_earthquake{r.event}_pre_{suf}.png"), pre[..., ::-1])
        cv2.imwrite(str(b / idr / f"turkey_earthquake{r.event}_post_{suf}.png"), post[..., ::-1])
        for c, m in masks.items():
            if m.any():                               # boş maske dosyası yazılmaz (gerçek veride de bazı sınıflar yok)
                d = b / "masks" / f"{pf}_building_{c}_mask"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{r.tile}.png"), m)
    val.merge(keep, on=["event", "tile"]).drop(columns=["split"]).to_csv(repo / cfg["split"]["validity"], index=False)
    print("sentetik veri:", data, "| sahte repo kökü:", repo, "| karo:", len(keep))


if __name__ == "__main__":
    main()
