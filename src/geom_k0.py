"""Adım 8B / K0 — geometri hızlı ölüm testi (kalıntı ofset). Ön-kayıt: docs/adim8b_onkayit.md §3-§4, docs/adim8b_k0.md.

Model eğitimi YOK. Her bina bileşeni (sınıf maskesi başına connectedComponents, 8-bağlantı) için:
  kontur = bileşen - erode(3x3); görüntü = şehir medyan kaymasıyla hizalanmış SONRASI görüntünün gri, Gauss(sigma=1), Sobel büyüklüğü;
  (dx,dy) in [-24,24]^2, adım 2: kontur piksellerinde gradyan toplamını maksimize eden öteleme (kalıntı ofset);
  proj = (dx,dy) . u,  u = şehir medyan kayma vektörünün birim yönü.
Hipotez işareti (sonuçlardan ÖNCE): alçalan bina daha az yaslanır -> proj küçük/negatif; K0 skoru = -proj, pozitif sınıf = hasarlı u yıkık.
Sınıf etiketi ofset hesabında kullanılmaz; AUC en sonda.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

R, STEP, MIN_AREA, MIN_VALID = 24, 2, 40, 0.9
CLS = ("intact", "damaged", "destroyed")


def gradient_map(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def building_offsets(G: np.ndarray, masks: dict, valid: np.ndarray, u: np.ndarray) -> list[dict]:
    """masks: {sınıf: ikili HxW}. Dönüş: bina başına sözlük listesi."""
    H, W = G.shape
    Gp = cv2.copyMakeBorder(G, R, R, R, R, cv2.BORDER_CONSTANT, value=0)
    k3 = np.ones((3, 3), np.uint8)
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
            cp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            cont = comp - cv2.erode(cp, k3)[1:-1, 1:-1]       # sıfır dolgu: kırpım kenarı bina sınırıdır
            if cont.sum() == 0:
                continue
            vcrop = valid[y:y + h, x:x + w]
            if (vcrop[cont > 0].mean()) < MIN_VALID:
                continue
            patch = Gp[y:y + h + 2 * R, x:x + w + 2 * R]          # Gp'de (x,y) -> (x+R,y+R): pencere [-R, +R]
            S = cv2.matchTemplate(patch, cont.astype(np.float32), cv2.TM_CCORR)   # (2R+1)x(2R+1)
            ii = np.arange(0, 2 * R + 1, STEP)
            Ss = S[np.ix_(ii, ii)]
            j, k = np.unravel_index(int(np.argmax(Ss)), Ss.shape)
            dy, dx = int(ii[j] - R), int(ii[k] - R)
            nc = float(cont.sum())
            rows.append(dict(cls=c, area=int(area), dx=dx, dy=dy, proj=float(dx * u[0] + dy * u[1]),
                             skor_tepe=float(Ss[j, k] / nc), skor_sifir=float(S[R, R] / nc)))
    return rows


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    s = pd.Series(score).rank(method="average").to_numpy()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def city_stats(df: pd.DataFrame, nboot: int = 2000, seed: int = 42) -> dict:
    """df: bina satırları (tile, pos[bool], proj, area). Skor = -proj."""
    sc, pos = -df.proj.to_numpy(), df.pos.to_numpy(bool)
    out = dict(n_bina=len(df), n_poz=int(pos.sum()), n_neg=int((~pos).sum()), auc=auc(sc, pos))
    tiles = df.tile.unique()
    groups = {t: g for t, g in df.groupby("tile")}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        pick = rng.choice(tiles, len(tiles), replace=True)
        b = pd.concat([groups[t] for t in pick])
        bs.append(auc(-b.proj.to_numpy(), b.pos.to_numpy(bool)))
    out["auc_ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    out["auc_isaretsiz"] = max(out["auc"], 1 - out["auc"])
    q = df.area.quantile([1 / 3, 2 / 3]).to_numpy()
    terc = np.digitize(df.area.to_numpy(), q)
    out["auc_alan_tertil"] = [auc(sc[terc == t], pos[terc == t]) if pos[terc == t].sum() and (~pos[terc == t]).sum() else float("nan") for t in range(3)]
    out["n_poz_tertil"] = [int(pos[terc == t].sum()) for t in range(3)]
    out["auc_oturma_tepe"] = auc(-df.skor_tepe.to_numpy(), pos)      # H1 kanıtı DEĞİL (doku kaybı)
    out["medyan_proj_neg"] = float(df.proj[~df.pos].median())
    out["medyan_proj_poz"] = float(df.proj[df.pos].median())
    return out


def run(root: Path, split_csv: Path, sample_csv: Path, out_dir: Path, control_pre: bool = True):
    from src.dm3_dataset import read_tile
    sp = pd.read_csv(split_csv)
    man = pd.read_csv(split_csv.parents[1] / "disasterm3_turkiye" / "manifest.csv", usecols=["event", "tile", "dx_post", "dy_post"])
    sp = sp.merge(man, on=["event", "tile"], how="left")
    smp = pd.read_csv(sample_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = []
    for g, gs in smp.groupby("group"):
        gall = sp[(sp.role == "train") & (sp.group == g)]
        s = (int(round(gall.dx_post.median())), int(round(gall.dy_post.median())))     # şehir medyan kayması (tüm şehir karoları)
        u = np.array(s, float) / np.linalg.norm(s)
        print(g, "şehir kayması", s, "|s|=%.1f" % np.linalg.norm(s))
        for _, r in gs.iterrows():
            pre, post, _, valid = read_tile(root, r.split, int(r.event), r.tile, shift=s)
            _, _, mp_ = __import__("src.dm3_dataset", fromlist=["tile_paths"]).tile_paths(root, r.split, int(r.event), r.tile)
            masks = {c: (cv2.imread(str(mp_[c]), cv2.IMREAD_GRAYSCALE) if mp_[c].exists() else None) for c in CLS}
            for img, tag in [(post, "post")] + ([(pre, "pre_kontrol")] if control_pre else []):
                for d in building_offsets(gradient_map(img), masks, valid, u):
                    d.update(group=g, tile=r.tile, goruntu=tag, pos=d["cls"] != "intact")
                    allrows.append(d)
        print(g, "bitti", sum(1 for d in allrows if d["group"] == g and d["goruntu"] == "post"), "bina")
    df = pd.DataFrame(allrows)
    df.to_csv(out_dir / "k0_binalar.csv.gz", index=False)
    res = {}
    for tag in df.goruntu.unique():
        for g in sorted(df.group.unique()):
            res[f"{tag}/{g}"] = city_stats(df[(df.goruntu == tag) & (df.group == g)])
    import json
    (out_dir / "k0_sonuc.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    a = [res[f"post/{g}"]["auc"] for g in sorted(df.group.unique())]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\nK0 KARARI:", "H1 DÜŞER (iki şehirde de AUC < 0,55)" if all(x < 0.55 for x in a) else "H1 düşmedi -> Kapı E koşulur",
          "| AUC:", dict(zip(sorted(df.group.unique()), np.round(a, 3))))
    return res


if __name__ == "__main__":
    from src.config import load
    from src.dm3_verify import find_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root")
    ap.add_argument("--no-control-pre", action="store_true")
    a = ap.parse_args()
    cfg = load(); base = Path(cfg["_root"])
    run(Path(a.root) if a.root else find_root(), base / cfg["split"]["S1"], base / "data_manifest/geom_k0/k0_orneklem_v1.csv",
        Path(a.out), control_pre=not a.no_control_pre)
