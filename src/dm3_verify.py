"""Adım 5 doğrulaması: Kaggle'a yüklenen DisasterM3 Türkiye dataset'indeki kullanılabilir karoları sayar
ve repodaki manifest.csv ile karşılaştırır (dosya adları üzerinden; görüntü açılmaz, saniyeler sürer).

    python -m src.dm3_verify                 # /kaggle/input altında bench/ + train/ içeren klasörü kendisi bulur
    python -m src.dm3_verify --root <klasör>  # ya da RAG_DATA ortam değişkeni

Çıkış kodu 0 = sayı beklenenle (config: data.usable_tiles_expected) ve manifest'le birebir aynı.
"""
import argparse, os, sys
from pathlib import Path

import pandas as pd

from src.config import load
from src.dm3_index import tiles_from_paths


def find_root(explicit=None):
    cands = [explicit, os.environ.get("RAG_DATA")]
    for c in cands:
        if c:
            return Path(c)
    base = Path("/kaggle/input")
    if base.exists():
        for dirpath, dirs, _ in os.walk(base):
            depth = len(Path(dirpath).relative_to(base).parts)
            if "bench" in dirs and "train" in dirs:
                return Path(dirpath)
            if depth >= 4:
                dirs[:] = []
    return None


def rel_files(split_dir: Path):
    for dp, _, fs in os.walk(split_dir):
        for f in fs:
            if os.path.getsize(os.path.join(dp, f)) > 0:
                yield Path(dp, f).relative_to(split_dir).as_posix()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    a = ap.parse_args(argv)
    cfg = load()
    root = find_root(a.root)
    if root is None or not root.exists():
        print("HATA: veri klasörü bulunamadı (dataset notebook'a eklendi mi? ya da --root / RAG_DATA ver)")
        return 2
    print("veri klasörü:", root)
    tiles = {}
    for split in ("train", "bench"):
        d = root / split
        n = sum(1 for _ in rel_files(d)) if d.exists() else 0
        print(f"  {split}: {n} dosya")
        tiles.update(tiles_from_paths(split, rel_files(d)) if d.exists() else {})
    ok = {k for k, v in tiles.items() if v["status"] == "ok"}
    exp = cfg["data"]["usable_tiles_expected"]
    print(f"kullanılabilir karo (öncesi+sonrası+maske): {len(ok)}   beklenen: {exp}")
    print(pd.Series([k[1] for k in sorted(ok)]).value_counts().sort_index().rename("karo").to_string())

    m = pd.read_csv(Path(cfg["_root"]) / cfg["data"]["manifest"])
    mok = {(r.split, int(r.event), r.tile) for r in m[m["status"] == "ok"].itertuples()}
    eksik, fazla = sorted(mok - ok), sorted(ok - mok)
    print(f"manifest ile karşılaştırma: eksik {len(eksik)}, fazla {len(fazla)}")
    for x in (eksik[:5] + fazla[:5]):
        print("   ", x)
    sar = sum(1 for k in ok if tiles[k]["sar"])
    print(f"bilgi: SAR dosyası olan karo: {sar}")
    good = len(ok) == exp and not eksik and not fazla
    print("SONUÇ:", "OK" if good else "SORUN VAR")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
