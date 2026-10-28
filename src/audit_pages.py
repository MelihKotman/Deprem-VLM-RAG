"""Adım 6: denetim sayfaları. Örneklemdeki her karo için öncesi ve sonrası görüntüyü bina konturlarıyla çizer.

    python -m src.audit_pages --out /kaggle/working/denetim [--root <veri>] [--quadrants] [--only A001,A002]

Üretilen (out/ altında):
  sayfalar/<audit_id>_pre.jpg   öncesi + kontur (yeşil=sağlam, sarı=hasarlı, kırmızı=yıkık); öncesi BOŞ (siyah) bölge macenta gölge
  sayfalar/<audit_id>_post.jpg  sonrası + aynı konturlar (ham etiket, hizalama YOK)
  sayfalar/<audit_id>_pre_q{1..4}.jpg  (--quadrants) öncesi, 2x yakınlaştırılmış çeyrekler
  sayfa_ozet.csv                audit_id, pre_blank_frac, etiketli piksel oranları (otomatik)
  tum_sayfalar.zip, dogrulama_sayfalari.zip (yalnız dogrulama==1 karolar)
Sayfa üstünde yalnız audit_id ve etiket bileşen sayıları yazar (şehir/event YAZILMAZ: körleme).
Izgara: 4x4 hücre (sütun A-D, satır 1-4), her hücre 256x256 özgün piksel; "C3" gibi konum notu için.
Yalnız numpy + Pillow + pandas kullanır.
"""
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import load
from src.dm3_verify import find_root

CLS = ("intact", "damaged", "destroyed")
COL = {"intact": (0, 255, 0), "damaged": (255, 255, 0), "destroyed": (255, 0, 0)}
SCALE = 1.25  # 1024 -> 1280


def paths(root: Path, split: str, ev: int, tile: str):
    suf = tile.split("_", 2)[2]
    if split == "train":
        b, idr, pf = root / "train", "train_images", "train"
    else:
        b, idr, pf = root / "bench", "test_images", "test"
    pre = b / idr / f"turkey_earthquake{ev}_pre_{suf}.png"
    post = b / idr / f"turkey_earthquake{ev}_post_{suf}.png"
    masks = {c: b / "masks" / f"{pf}_building_{c}_mask" / f"{tile}.png" for c in CLS}
    return pre, post, masks


def edge(m):
    e = m.copy()
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    e[0, :] = m[0, :]; e[-1, :] = m[-1, :]; e[:, 0] = m[:, 0]; e[:, -1] = m[:, -1]
    return e


def dilate(e):
    d = e.copy()
    d[1:] |= e[:-1]; d[:-1] |= e[1:]; d[:, 1:] |= e[:, :-1]; d[:, :-1] |= e[:, 1:]
    return d


def render(img, masks, blank, header, shade_blank):
    """img: HxWx3 uint8 (1024). -> PIL.Image (SCALE ile büyütülmüş, kontur+ızgara+başlık)."""
    n = int(round(img.shape[0] * SCALE))
    base = Image.fromarray(img).resize((n, n), Image.LANCZOS)
    a = np.array(base).astype(np.float32)
    if shade_blank and blank is not None and blank.any():
        bm = np.array(Image.fromarray((blank * 255).astype(np.uint8)).resize((n, n), Image.NEAREST)) > 0
        a[bm] = 0.55 * a[bm] + 0.45 * np.array([255, 0, 255], np.float32)
    a = a.astype(np.uint8)
    for c in CLS:
        m = masks[c]
        if m is None or not m.any():
            continue
        mm = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((n, n), Image.NEAREST)) > 0
        a[edge(mm)] = COL[c]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    step = n // 4
    for i in range(1, 4):
        for t in range(0, n, 12):  # kesikli ızgara
            d.line([(i * step, t), (i * step, t + 5)], fill=(255, 255, 255), width=1)
            d.line([(t, i * step), (t + 5, i * step)], fill=(255, 255, 255), width=1)
    for r in range(4):
        for cc in range(4):
            d.text((cc * step + 4, r * step + 2), f"{'ABCD'[cc]}{r + 1}", fill=(255, 255, 255))
    out = Image.new("RGB", (n, n + 28), (20, 20, 20))
    out.paste(im, (0, 28))
    ImageDraw.Draw(out).text((6, 8), header, fill=(255, 255, 255))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root")
    ap.add_argument("--quadrants", action="store_true")
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    cfg = load()
    root = find_root(a.root)
    if root is None:
        raise SystemExit("veri klasörü bulunamadı")
    s = pd.read_csv(Path(cfg["_root"]) / "data_manifest/denetim/denetim_orneklem_v1.csv")
    if a.only:
        s = s[s.audit_id.isin(a.only.split(","))]
    out = Path(a.out); pg = out / "sayfalar"; pg.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in s.itertuples():
        pre_p, post_p, mp = paths(root, r.split, int(r.event), r.tile)
        pre = np.array(Image.open(pre_p).convert("RGB")); post = np.array(Image.open(post_p).convert("RGB"))
        masks = {c: (np.array(Image.open(p)) > 0) if p.exists() else None for c, p in mp.items()}
        blank = pre.sum(axis=2) == 0
        px = {c: (int(masks[c].sum()) if masks[c] is not None else 0) for c in CLS}
        head = f"{r.audit_id}  etiketli bilesen: sagl {r.n_intact} / hasarli {r.n_damaged} / yikik {r.n_destroyed}"
        im = render(pre, masks, blank, head + "  [ONCESI]", True); im.save(pg / f"{r.audit_id}_pre.jpg", quality=88)
        render(post, masks, None, head + "  [SONRASI]", False).save(pg / f"{r.audit_id}_post.jpg", quality=88)
        if a.quadrants:
            full = render(pre, masks, blank, head, True).crop((0, 28, 1280, 1308))
            h = 640
            for qi, (x, y) in enumerate([(0, 0), (h, 0), (0, h), (h, h)], 1):
                q = full.crop((x, y, x + h, y + h)).resize((1024, 1024), Image.LANCZOS)
                q.save(pg / f"{r.audit_id}_pre_q{qi}.jpg", quality=88)
        rows.append(dict(audit_id=r.audit_id, pre_blank_frac=round(float(blank.mean()), 4),
                         **{f"px_{c}": px[c] for c in CLS}))
    pd.DataFrame(rows).to_csv(out / "sayfa_ozet.csv", index=False)
    ver = set(s[s.dogrulama == 1].audit_id)
    with zipfile.ZipFile(out / "tum_sayfalar.zip", "w") as z1, zipfile.ZipFile(out / "dogrulama_sayfalari.zip", "w") as z2:
        for f in sorted(pg.glob("*.jpg")):
            z1.write(f, f.name)
            if f.name.split("_")[0] in ver:
                z2.write(f, f.name)
    print("sayfa:", len(rows), "karo;", out)


if __name__ == "__main__":
    main()
