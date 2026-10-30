"""Adım 8B / K1 — gölge denemesi. Ön-kayıt: docs/adim8b_k1.md (sonuçtan önce yazıldı). Model eğitimi YOK.

Her bina için 16 yönde, 4..32 px aralığında (ötelenmiş taban izi) zeminin yerel zemine göre ne kadar karardığı (C_t) ölçülür.
Gölge yönü şehir başına SINIF ETİKETİ KULLANILMADAN seçilir. Skor = -S(yön*); pozitif = hasarlı u yıkık. Tabakalı (alan tertili) AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.geom_k0 import auc

CLS = ("intact", "damaged", "destroyed")
MARG, MIN_AREA, MIN_VALID = 36, 40, 0.9
T = np.arange(4, 33, 4)
NDIR = 16
ANG = np.deg2rad(np.arange(NDIR) * 22.5)
UX, UY = np.cos(ANG), np.sin(ANG)
SHIFTS = [[(int(round(t * UX[k])), int(round(t * UY[k]))) for t in T] for k in range(NDIR)]


def prep(post_rgb, masks, valid):
    gray = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    allb = np.zeros(gray.shape, np.uint8)
    for m in masks.values():
        if m is not None:
            allb |= (m > 0).astype(np.uint8)
    allb = cv2.dilate(allb, np.ones((7, 7), np.uint8))
    ok = (valid & (allb == 0)).astype(np.uint8)
    G = cv2.copyMakeBorder(gray, MARG, MARG, MARG, MARG, cv2.BORDER_CONSTANT, value=0)
    OK = cv2.copyMakeBorder(ok, MARG, MARG, MARG, MARG, cv2.BORDER_CONSTANT, value=0).astype(bool)
    return G, OK


def building_S(G, OK, x, y, comp):
    """comp: bina bileşeni (h x w, uint8); (x,y) orijinal karodaki sol-üst. -> S[16] (nan olabilir)."""
    h, w = comp.shape
    Gc = G[y:y + h + 2 * MARG, x:x + w + 2 * MARG]
    Oc = OK[y:y + h + 2 * MARG, x:x + w + 2 * MARG]
    if Oc.sum() < 200:
        return np.full(NDIR, np.nan)
    bg = float(np.percentile(Gc[Oc], 75))      # gölgeye dayanıklı zemin parlaklığı (medyan, binanın kendi gölgesinden etkileniyordu)
    if bg <= 1:
        return np.full(NDIR, np.nan)
    ys, xs = np.nonzero(comp)
    ys, xs = ys + MARG, xs + MARG
    S = np.full(NDIR, np.nan)
    for k in range(NDIR):
        C = []
        for sx, sy in SHIFTS[k]:
            yy, xx = ys + sy, xs + sx
            ok = Oc[yy, xx]
            if ok.sum() >= 30:
                C.append(1.0 - Gc[yy, xx][ok].mean() / bg)
        if len(C) >= 4:
            S[k] = float(np.mean(C))
    return S


def tile_rows(post_rgb, masks, valid):
    G, OK = prep(post_rgb, masks, valid)
    rows = []
    for c, m in masks.items():
        if m is None:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < MIN_AREA:
                continue
            comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            if valid[y:y + h, x:x + w][comp > 0].mean() < MIN_VALID:
                continue
            S = building_S(G, OK, x, y, comp)
            if np.isfinite(S).sum() == 0:
                continue
            rows.append(dict(cls=c, area=int(area), **{f"s{k:02d}": S[k] for k in range(NDIR)}))
    return rows


def choose_dir(df: pd.DataFrame):
    mu = np.array([np.nanmean(df[f"s{k:02d}"].to_numpy()) for k in range(NDIR)])
    return int(np.argmax(mu)), float(mu.max() - np.median(mu)), mu


def strat_auc(score, pos, terc):
    num = den = 0.0
    per = []
    for t in range(3):
        m = terc == t
        n1, n0 = int(pos[m].sum()), int((~pos[m]).sum())
        if n1 == 0 or n0 == 0:
            per.append(float("nan")); continue
        a = auc(score[m], pos[m]); per.append(a)
        num += a * n1 * n0; den += n1 * n0
    return (num / den if den else float("nan")), per


def city_stats(df: pd.DataFrame, nboot=1000, seed=42):
    k, sharp, mu = choose_dir(df)
    d = df.copy()
    d["S"] = d[f"s{k:02d}"]; d["Sctl"] = d[f"s{(k + 8) % NDIR:02d}"]
    d = d[np.isfinite(d.S) & np.isfinite(d.Sctl)].reset_index(drop=True)
    pos = (d.cls != "intact").to_numpy()
    q = d.area.quantile([1 / 3, 2 / 3]).to_numpy()
    terc = np.digitize(d.area.to_numpy(), q)
    sc = -d.S.to_numpy()
    a, per = strat_auc(sc, pos, terc)
    tiles = d.tile.unique(); by = {t: np.flatnonzero((d.tile == t).to_numpy()) for t in tiles}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        idx = np.concatenate([by[t] for t in rng.choice(tiles, len(tiles), replace=True)])
        bs.append(strat_auc(sc[idx], pos[idx], terc[idx])[0])
    out = dict(yon_idx=k, yon_derece=k * 22.5, keskinlik=sharp, n_bina=len(d), n_poz=int(pos.sum()),
               tabakali_auc=a, ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
               tertil_auc=per, tertil_n_poz=[int(pos[terc == t].sum()) for t in range(3)],
               kontrol_ters_yon_auc=strat_auc(-d.Sctl.to_numpy(), pos, terc)[0],
               isaretsiz_auc=max(a, 1 - a) if np.isfinite(a) else float("nan"))
    de = (d.cls == "destroyed").to_numpy(); keep = de | (d.cls == "intact").to_numpy()
    out["yikik_vs_saglam_tabakali_auc"] = strat_auc(sc[keep], de[keep], terc[keep])[0]
    sg = (d.cls == "intact").to_numpy()
    out["saglam_alan_S_spearman"] = float(np.corrcoef(pd.Series(d.area[sg].to_numpy()).rank().to_numpy(), pd.Series(d.S[sg].to_numpy()).rank().to_numpy())[0, 1])
    out["yon_ortalamalari"] = [float(v) for v in mu]
    return out


def run(root: Path, split_csv: Path, out_dir: Path):
    from src.dm3_dataset import read_tile, tile_paths
    sp = pd.read_csv(split_csv)
    man = pd.read_csv(split_csv.parents[1] / "disasterm3_turkiye" / "manifest.csv", usecols=["event", "tile", "dx_post", "dy_post"])
    sp = sp.merge(man, on=["event", "tile"], how="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = []
    for g in ["Adiyaman", "Malatya"]:
        gall = sp[(sp.role == "train") & (sp.group == g)]
        s = (int(round(gall.dx_post.median())), int(round(gall.dy_post.median())))
        print(g, "şehir kayması", s, "karo", len(gall), flush=True)
        for j, (_, r) in enumerate(gall.iterrows()):
            _, post, _, valid = read_tile(root, r.split, int(r.event), r.tile, shift=s)
            _, _, mp_ = tile_paths(root, r.split, int(r.event), r.tile)
            masks = {c: (cv2.imread(str(mp_[c]), cv2.IMREAD_GRAYSCALE) if mp_[c].exists() else None) for c in CLS}
            for d in tile_rows(post, masks, valid):
                d.update(group=g, tile=r.tile); allrows.append(d)
            if j % 25 == 0:
                print(g, j, "/", len(gall), "bina", sum(1 for d in allrows if d["group"] == g), flush=True)
    df = pd.DataFrame(allrows); df.to_csv(out_dir / "k1_binalar.csv.gz", index=False)
    res = {g: city_stats(df[df.group == g]) for g in ["Adiyaman", "Malatya"]}
    (out_dir / "k1_sonuc.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({g: {k: v for k, v in r.items() if k != "yon_ortalamalari"} for g, r in res.items()}, indent=1, ensure_ascii=False))
    ok_meas = all(r["keskinlik"] >= 0.01 for r in res.values())
    gate = all(r["tabakali_auc"] >= 0.60 and r["ci95"][0] > 0.55 for r in res.values())
    print("\nK1 KARARI:", "ÖLÇÜLEMEZ (gölge yönü keskinliği < 0,01)" if not ok_meas else ("GEÇTİ -> (iii) ön-kaydı yazılır" if gate else "GEÇMEDİ -> geometri kolu KAPANIR"))
    return res


if __name__ == "__main__":
    from src.config import load
    from src.dm3_verify import find_root
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--root")
    a = ap.parse_args(); cfg = load()
    run(Path(a.root) if a.root else find_root(), Path(cfg["_root"]) / cfg["split"]["S1"], Path(a.out))
