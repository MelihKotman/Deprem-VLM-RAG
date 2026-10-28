"""Adım 7: S1/S2 bölme dosyalarını manifest'ten üretir (olay düzeyinde, deterministik).

    python -m src.make_splits                 # data_manifest/splits/split_S{1,2}_v1.csv yazar + özet basar
    python -m src.make_splits --check         # var olan dosyaları yeniden üretip birebir aynı mı diye bakar (dondurma denetimi)

Kural: bölme olay (event) düzeyinde; zip'e (bench/train) göre DEĞİL. Rol: train | test | control.
Sızıntı kontrolü (Kaggle, src/leak_check.py) sonucu bir olay bir şehirle örtüşüyorsa o olay
`data_manifest/splits/leak_overrides_v1.csv` ile örtüştüğü şehirle aynı tarafa alınır; dosya yoksa/boşsa değişiklik yok.
Kayma kademesi (kayma_tier) manifest'teki kayma_post_px (otomatik tahmin) üzerinden, Adım 6 denetim kademeleriyle uyumlu:
  T0 <= 5 px, T1 (5,15], T2 (15,30], T3 > 30, TX = tahmin yok.
"""
import argparse, hashlib, sys
from pathlib import Path

import pandas as pd

from src.config import load

VERSION = "v1"
EVENT_NAME = {0: "Gaziantep", 1: "Adiyaman", 3: "Malatya", 4: "Antakya/Defne", 5: "Osmaniye",
              6: "bilinmiyor(event6)", 7: "bilinmiyor(event7)"}
SCENARIOS = {
    "S1": {"train": [1, 3, 5], "test": [4, 6, 7], "control": [0]},
    "S2": {"train": [1, 3, 4, 6], "test": [5, 7], "control": [0]},
}
TIER_BINS = [-1, 5, 15, 30, float("inf")]
TIER_LABELS = ["T0", "T1", "T2", "T3"]


def load_overrides(d: Path) -> dict:
    p = d / f"leak_overrides_{VERSION}.csv"
    if not p.exists():
        return {}
    o = pd.read_csv(p)
    return {(r.scenario, int(r.event)): r.role for r in o.itertuples()}


def build(manifest: pd.DataFrame, overrides: dict) -> dict:
    ok = manifest[manifest["status"] == "ok"].copy()
    ok["kayma_tier"] = pd.cut(ok["kayma_post_px"], TIER_BINS, labels=TIER_LABELS).astype(object)
    ok["kayma_tier"] = ok["kayma_tier"].where(ok["kayma_post_px"].notna(), "TX")
    ok["source_known"] = ok["event"].isin([0, 1, 3, 4, 5])
    out = {}
    for sc, roles in SCENARIOS.items():
        role_of = {ev: r for r, evs in roles.items() for ev in evs}
        for (s, ev), r in overrides.items():
            if s == sc:
                role_of[ev] = r
        df = ok.copy()
        df["role"] = df["event"].map(role_of)
        assert df["role"].notna().all(), f"{sc}: rolü olmayan olay"
        df["scenario"] = sc
        df["group"] = df["event"].map(EVENT_NAME)
        cols = ["scenario", "role", "event", "group", "split", "tile", "source_known", "kayma_post_px",
                "kayma_tier", "n_intact", "n_damaged", "n_destroyed", "has_sar", "off_nadir"]
        out[sc] = df[cols].sort_values(["role", "event", "tile"]).reset_index(drop=True)
    return out


def sha1_of(df: pd.DataFrame) -> str:
    return hashlib.sha1(df.to_csv(index=False).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = load()
    root = Path(cfg["_root"])
    man = pd.read_csv(root / cfg["data"]["manifest"])
    d = root / "data_manifest" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    res = build(man, load_overrides(d))
    rc = 0
    for sc, df in res.items():
        p = d / f"split_{sc}_{VERSION}.csv"
        if a.check:
            same = p.exists() and p.read_text() == df.to_csv(index=False)
            print(f"{sc}: {'AYNI' if same else 'FARKLI'}  {p.name}")
            rc |= 0 if same else 1
        else:
            df.to_csv(p, index=False)
            print(f"yazıldı {p.relative_to(root)}  satır={len(df)}  sha1={sha1_of(df)[:10]}")
        print(df.groupby(["role", "event", "group"]).size().rename("karo").to_string())
        print("  rol toplamları:", df["role"].value_counts().to_dict())
        print(pd.crosstab([df["role"], df["event"]], df["kayma_tier"]).to_string())
        ev_roles = df.groupby("event")["role"].nunique()
        assert (ev_roles == 1).all(), "olay iki role bölünmüş (sızıntı)"
        assert df["tile"].is_unique, "karo tekrarı"
    return rc


if __name__ == "__main__":
    sys.exit(main())
