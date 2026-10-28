"""Adım 7: event'ler arası coğrafi örtüşme (sızıntı) kontrolü. Kaggle'da çalışır (görüntü gerekir, CPU yeter).

    python -m src.leak_check --out /kaggle/working/leak [--root <veri>] [--workers 4] [--also-post]

Yöntem: her karonun ÖNCESİ görüntüsünden (boş/siyah bölge dışında) SIFT özellikleri çıkarılır; farklı event'lerden her karo
çifti için Lowe oran testi + RANSAC homografi ile "tutarlı eşleşme" (inlier) sayısı bulunur. Aynı yeri farklı kaydırma,
ölçek veya çekim tarihiyle gösteren iki karo çok sayıda geometrik tutarlı eşleşme verir; farklı yerler vermez.
(--also-post: aynısı SONRASI görüntüler için de.)

ÖN-KAYIT (28 Ekim 2026, hiçbir çift skorlanmadan ÖNCE yazıldı; sonuca bakıp değiştirilmez):
  - Eşleşen karo çifti  : inlier >= 30  VE  inlier / iyi_eşleşme >= 0.30   (MATCH_INLIERS, MATCH_RATIO)
  - Event çifti "ÖRTÜŞÜYOR" : >= 3 eşleşen karo çifti;  1-2 çift "ŞÜPHELİ" (elle bakılır, yan yana görsel üretilir);  0 = "TEMİZ"
  - Kontrol: aynı karonun öncesi-sonrası eşleşmesi (pozitif; yöntem çalışıyor mu) ve karonun kırpılıp ölçeklenmiş kopyası (pozitif).
    Pozitif kontrollerde eşleşme oranı < %80 ise yöntem yetersiz sayılır ve sonuç "KONTROLSÜZ" yazılır.
  - Karar: örtüşen event bilinen şehirle aynı tarafa konur (bkz. src/make_splits.py, leak_overrides_v1.csv).
Çıktılar (out/): eslesen_ciftler.csv, event_matrisi.csv, kontrol.csv, ozet.txt, gorseller/*.jpg
"""
import argparse, itertools, multiprocessing as mp, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MATCH_INLIERS = 30
MATCH_RATIO = 0.30
LOW_FLOOR = 12          # bu altı kaydedilmez
SIZE = 512              # özellik çıkarma boyutu
NFEAT = 600
LOWE = 0.75
OVERLAP_MIN_PAIRS = 3
POS_CONTROL_MIN = 0.80

_D = {}      # global: tile -> (kp_xy float32 Nx2, desc float32 NxD)


def img_paths(root: Path, split: str, ev: int, tile: str):
    suf = tile.split("_", 2)[2]
    b, idr = (root / "train", "train_images") if split == "train" else (root / "bench", "test_images")
    return (b / idr / f"turkey_earthquake{ev}_pre_{suf}.png", b / idr / f"turkey_earthquake{ev}_post_{suf}.png")


def feats(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return feats_from_array(im)


def feats_from_array(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    valid = (im.sum(axis=2) > 0).astype(np.uint8) * 255
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kp, d = sift.detectAndCompute(g, valid)
    if d is None or len(kp) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], np.float32), d.astype(np.float32)


def score(a, b):
    """(inlier, iyi_eşleşme) iki özellik kümesi arasında."""
    (pa, da), (pb, db) = a, b
    if len(da) < 8 or len(db) < 8:
        return 0, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < LOWE * x[1].distance]
    if len(good) < 8:
        return 0, len(good)
    src = pa[[g.queryIdx for g in good]].reshape(-1, 1, 2)
    dst = pb[[g.trainIdx for g in good]].reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return (int(inl.sum()) if inl is not None else 0), len(good)


def _work(pair):
    i, j = pair
    s, g = score(_D[i], _D[j])
    return i, j, s, g


def run_pairs(keys, ev, workers):
    pairs = [(i, j) for i, j in itertools.combinations(range(len(keys)), 2) if ev[i] != ev[j]]
    out = []
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as p:
        for n, r in enumerate(p.imap_unordered(_work, pairs, chunksize=400), 1):
            if r[2] >= LOW_FLOOR:
                out.append(r)
            if n % 50000 == 0:
                print(f"  {n}/{len(pairs)} çift, {time.time() - t0:.0f} sn", flush=True)
    return pairs, out


def is_match(s, g):
    return s >= MATCH_INLIERS and g > 0 and s / g >= MATCH_RATIO


def verdict(n):
    return "ÖRTÜŞÜYOR" if n >= OVERLAP_MIN_PAIRS else ("ŞÜPHELİ" if n >= 1 else "TEMİZ")


def controls(root, tiles, rng, n=40):
    rows = []
    pick = rng.choice(len(tiles), size=min(n, len(tiles)), replace=False)
    for k in pick:
        r = tiles.iloc[k]
        pre_p, post_p = img_paths(root, r.split, int(r.event), r.tile)
        pre = cv2.imread(str(pre_p)); post = cv2.imread(str(post_p))
        if pre is None or post is None:
            continue
        fa = feats_from_array(pre)
        s1, g1 = score(fa, feats_from_array(post))
        # kırpılmış (%60) + 1.3x ölçek + 10° dönme
        h, w = pre.shape[:2]
        y0, x0 = int(rng.integers(0, h // 3)), int(rng.integers(0, w // 3))
        crop = pre[y0:y0 + int(h * 0.6), x0:x0 + int(w * 0.6)]
        M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), 10, 1.3)
        crop = cv2.warpAffine(crop, M, (w, h))
        s2, g2 = score(fa, feats_from_array(crop))
        rows.append(dict(event=int(r.event), tile=r.tile, pre_post_inlier=s1, pre_post_good=g1,
                         crop_inlier=s2, crop_good=g2, pre_post_ok=is_match(s1, g1), crop_ok=is_match(s2, g2)))
    return pd.DataFrame(rows)


def side_by_side(root, ta, tb, path, label):
    a = cv2.imread(str(img_paths(root, ta.split, int(ta.event), ta.tile)[0]))
    b = cv2.imread(str(img_paths(root, tb.split, int(tb.event), tb.tile)[0]))
    if a is None or b is None:
        return
    a = cv2.resize(a, (512, 512)); b = cv2.resize(b, (512, 512))
    im = np.hstack([a, b])
    cv2.putText(im, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root")
    ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--also-post", action="store_true")
    ap.add_argument("--limit", type=int, help="deneme için: her event'ten en çok N karo")
    a = ap.parse_args(argv)
    from src.config import load
    from src.dm3_verify import find_root
    cfg = load()
    root = Path(a.root) if a.root else find_root()
    man = pd.read_csv(a.manifest or (Path(cfg["_root"]) / cfg["data"]["manifest"]))
    tiles = man[man.status == "ok"].reset_index(drop=True)
    if a.limit:
        tiles = tiles.groupby("event", group_keys=False).head(a.limit).reset_index(drop=True)
    out = Path(a.out); (out / "gorseller").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    ctrl = controls(root, tiles, rng)
    ctrl.to_csv(out / "kontrol.csv", index=False)
    pos_ok = float(ctrl[["pre_post_ok", "crop_ok"]].values.mean()) if len(ctrl) else 0.0
    print(f"pozitif kontrol başarı oranı: {pos_ok:.2f} (en az {POS_CONTROL_MIN} gerekir)")

    ev = tiles.event.astype(int).tolist()
    summary = []
    for kind, idx in (("pre", 0),) + ((("post", 1),) if a.also_post else ()):
        print(f"[{kind}] özellik çıkarılıyor ({len(tiles)} karo)...", flush=True)
        _D.clear()
        for i, r in tiles.iterrows():
            f = feats(img_paths(root, r.split, int(r.event), r.tile)[idx])
            _D[i] = f if f is not None else (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        pairs, res = run_pairs(list(range(len(tiles))), ev, a.workers)
        rows = []
        for i, j, s, g in res:
            rows.append(dict(kind=kind, event_a=ev[i], tile_a=tiles.tile[i], event_b=ev[j], tile_b=tiles.tile[j],
                             inlier=s, good=g, ratio=round(s / g, 3) if g else 0.0, match=is_match(s, g)))
        df = pd.DataFrame(rows)
        df.to_csv(out / f"eslesen_ciftler_{kind}.csv", index=False)
        m = df[df.match] if len(df) else df
        evs = sorted(set(ev))
        for x, y in itertools.combinations(evs, 2):
            n = int(((m.event_a == x) & (m.event_b == y) | (m.event_a == y) & (m.event_b == x)).sum()) if len(m) else 0
            mx = int(df[((df.event_a == x) & (df.event_b == y)) | ((df.event_a == y) & (df.event_b == x))].inlier.max()) \
                if len(df) and len(df[((df.event_a == x) & (df.event_b == y)) | ((df.event_a == y) & (df.event_b == x))]) else 0
            summary.append(dict(kind=kind, event_a=x, event_b=y, eslesen_karo_cifti=n, max_inlier=mx,
                                karar=("KONTROLSÜZ" if pos_ok < POS_CONTROL_MIN else verdict(n))))
            if 1 <= n and len(m):
                sub = m[((m.event_a == x) & (m.event_b == y)) | ((m.event_a == y) & (m.event_b == x))].nlargest(3, "inlier")
                for _, r in sub.iterrows():
                    ta = tiles[tiles.tile == r.tile_a].iloc[0]; tb = tiles[tiles.tile == r.tile_b].iloc[0]
                    side_by_side(root, ta, tb, out / "gorseller" / f"{kind}_e{x}_e{y}_{r.tile_a}__{r.tile_b}.jpg",
                                 f"{kind} e{ta.event} {ta.tile} | e{tb.event} {tb.tile}  inlier={r.inlier}")
    S = pd.DataFrame(summary)
    S.to_csv(out / "event_matrisi.csv", index=False)
    with open(out / "ozet.txt", "w") as f:
        f.write(f"pozitif kontrol: {pos_ok:.2f}\n{S.to_string(index=False)}\n")
    print(S.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
