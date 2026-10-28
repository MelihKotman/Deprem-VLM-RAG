"""Adım 7: değerlendirme geçerlilik maskesi (öncesi veya sonrası görüntüsü BOŞ olan bölgeleri değerlendirme dışı bırakır).

    python -m src.tile_validity --out /kaggle/working/validity [--root <veri>] [--save-masks]

Kural (28 Ekim 2026, sonuca bakmadan yazıldı; Adım 6 devri C1):
  boş(öncesi) = üç kanal toplamı 0 olan piksel; boş(sonrası) aynı. Küçük delikler kapatılır (5x5 kapama).
  eval_gecerli = ~boş(öncesi) & ~boş(sonrası), sonra 3 px aşındırılır (kenar artefaktı).
  Değerlendirmede (bina düzeyi ve karo düzeyi) tahminler ve referans yalnız eval_gecerli içinde sayılır; tamamı dışarıda kalan
  bina bileşeni değerlendirmeye girmez. Karo ATILMAZ (yalnız piksel maskelenir); gecerli_oran < 0.5 olan karolar `agir_bos=1`
  bayrağıyla ayrıca raporlanır (duyarlılık analizi için), dışarıda bırakılmaz.
Çıktılar: out/tile_validity_v1.csv (repoya kopyalanır: data_manifest/splits/), out/ozet.csv, --save-masks ile out/masks/<karo>.png (0/255)
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

AGIR_BOS = 0.5
CLS = ("intact", "damaged", "destroyed")


def valid_mask(pre, post):
    k5 = np.ones((5, 5), np.uint8)
    bp = (pre.sum(axis=2) == 0).astype(np.uint8)
    bq = (post.sum(axis=2) == 0).astype(np.uint8)
    bp = cv2.morphologyEx(bp, cv2.MORPH_CLOSE, k5); bq = cv2.morphologyEx(bq, cv2.MORPH_CLOSE, k5)
    ok = ((bp == 0) & (bq == 0)).astype(np.uint8)
    ok = cv2.erode(ok, np.ones((7, 7), np.uint8))   # ~3 px
    return ok.astype(bool), bp.astype(bool), bq.astype(bool)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root")
    ap.add_argument("--manifest")
    ap.add_argument("--save-masks", action="store_true")
    a = ap.parse_args(argv)
    from src.config import load
    from src.dm3_verify import find_root
    from src.audit_pages import paths
    cfg = load()
    root = Path(a.root) if a.root else find_root()
    man = pd.read_csv(a.manifest or (Path(cfg["_root"]) / cfg["data"]["manifest"]))
    tiles = man[man.status == "ok"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.save_masks:
        (out / "masks").mkdir(exist_ok=True)
    rows = []
    for r in tiles.itertuples():
        pre_p, post_p, mp_ = paths(root, r.split, int(r.event), r.tile)
        pre = cv2.imread(str(pre_p)); post = cv2.imread(str(post_p))
        if pre is None or post is None:
            rows.append(dict(event=r.event, tile=r.tile, hata=1)); continue
        ok, bp, bq = valid_mask(pre, post)
        lab = {}
        for c in CLS:
            m = cv2.imread(str(mp_[c]), cv2.IMREAD_GRAYSCALE) if mp_[c].exists() else None
            lab[c] = (m > 0) if m is not None else np.zeros(ok.shape, bool)
        tot = sum(int(v.sum()) for v in lab.values())
        inval = sum(int((v & ~ok).sum()) for v in lab.values())
        rows.append(dict(event=int(r.event), tile=r.tile, hata=0, bos_oncesi=round(float(bp.mean()), 4),
                         bos_sonrasi=round(float(bq.mean()), 4), gecerli_oran=round(float(ok.mean()), 4),
                         etiket_px=tot, etiket_px_gecersiz=inval,
                         etiket_gecersiz_oran=round(inval / tot, 4) if tot else 0.0,
                         agir_bos=int(ok.mean() < AGIR_BOS)))
        if a.save_masks:
            cv2.imwrite(str(out / "masks" / f"{r.tile}.png"), ok.astype(np.uint8) * 255)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tile_validity_v1.csv", index=False)
    g = df[df.hata == 0].groupby("event").agg(karo=("tile", "size"), gecerli_ort=("gecerli_oran", "mean"),
                                              agir_bos=("agir_bos", "sum"), bos_oncesi_ort=("bos_oncesi", "mean"),
                                              etiket_gecersiz_ort=("etiket_gecersiz_oran", "mean")).round(4)
    g.to_csv(out / "ozet.csv")
    print(g.to_string())
    print("hata:", int(df.hata.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
