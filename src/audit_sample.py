"""Adım 6: tabakalı denetim örneklemi (ÖNCEDEN KAYITLI; görüntülere bakmadan üretilir).

Kurallar (notlar/denetim-sonucu.md, protokol bölümü):
  - Şehir/event başına karo sayısı sabit: Antakya 22, e6 22, e7 12, Adıyaman 12, Malatya 12, Osmaniye 12, Gaziantep 4 (=96).
  - Şehir içinde hasar durumu tabakaları: D (en az 1 yıkık bina), H (yıkık yok, hasarlı var), S (yalnız sağlam).
    Orantılı dağıtım (en büyük kalan), dolu her tabakadan en az 1 karo.
  - Rastgelelik: seed 42 (örnekleme), 44 (denetim kimliği karıştırma), 43 (doğrulama alt kümesi).
  - Ağırlık = N_tabaka / n_tabaka (tabaka = şehir x hasar durumu).
Çıktı: data_manifest/denetim/denetim_orneklem_v1.csv  (üretildikten sonra DEĞİŞTİRİLMEZ)
    python -m src.audit_sample
"""
import hashlib
import math
import random
from pathlib import Path

import pandas as pd

from src.config import load

ALLOC = {"Antakya/Defne": 22, "bilinmiyor(event6)": 22, "bilinmiyor(event7)": 12,
         "Adiyaman": 12, "Malatya": 12, "Osmaniye": 12, "Gaziantep": 4}
VERIF = {"Antakya/Defne": 6, "bilinmiyor(event6)": 6, "bilinmiyor(event7)": 3,
         "Adiyaman": 3, "Malatya": 3, "Osmaniye": 3, "Gaziantep": 0}
S1_ROL = {"Antakya/Defne": "test", "bilinmiyor(event6)": "test", "bilinmiyor(event7)": "test",
          "Adiyaman": "egitim", "Malatya": "egitim", "Osmaniye": "egitim", "Gaziantep": "kontrol"}
S2_ROL = {"Antakya/Defne": "egitim", "bilinmiyor(event6)": "egitim", "bilinmiyor(event7)": "test",
          "Adiyaman": "egitim", "Malatya": "egitim", "Osmaniye": "test", "Gaziantep": "kontrol"}
SEED_SAMPLE, SEED_VERIF, SEED_ID = 42, 43, 44


def alloc(counts: dict, total: int) -> dict:
    """Orantılı dağıtım, en büyük kalan; dolu tabakaya en az 1; tabaka boyutunu aşmaz."""
    N = sum(counts.values())
    raw = {k: total * v / N for k, v in counts.items()}
    a = {k: min(counts[k], max(1, math.floor(raw[k]))) for k in counts}
    while sum(a.values()) < total:
        c = [k for k in counts if a[k] < counts[k]]
        k = max(c, key=lambda k: raw[k] - a[k])
        a[k] += 1
    while sum(a.values()) > total:
        c = [k for k in counts if a[k] > 1]
        k = min(c, key=lambda k: raw[k] - a[k])
        a[k] -= 1
    return a


def main():
    cfg = load()
    m = pd.read_csv(Path(cfg["_root"]) / cfg["data"]["manifest"])
    k = m[m.status == "ok"].copy()
    k["dur"] = k.apply(lambda r: "D" if r.n_destroyed > 0 else ("H" if r.n_damaged > 0 else "S"), axis=1)
    k["n_labeled"] = k.n_intact + k.n_damaged + k.n_destroyed
    rng = random.Random(SEED_SAMPLE)
    parts = []
    for city, n in ALLOC.items():
        c = k[k.bolge == city].sort_values("tile")
        counts = c.dur.value_counts().to_dict()
        a = alloc(counts, n)
        for d in sorted(counts):
            g = c[c.dur == d]
            idx = sorted(rng.sample(list(g.index), a[d]))
            s = g.loc[idx].copy()
            s["N_tabaka"], s["n_tabaka"] = counts[d], a[d]
            parts.append(s)
    s = pd.concat(parts)
    s["weight"] = s.N_tabaka / s.n_tabaka
    s["S1_rol"] = s.bolge.map(S1_ROL)
    s["S2_rol"] = s.bolge.map(S2_ROL)
    # karıştırılmış, bilgi taşımayan kimlik
    ids = list(s.index)
    random.Random(SEED_ID).shuffle(ids)
    s["audit_id"] = pd.Series({i: f"A{j + 1:03d}" for j, i in enumerate(ids)})
    # doğrulama alt kümesi (Melih'in bağımsız sayacağı)
    rv = random.Random(SEED_VERIF)
    s["dogrulama"] = 0
    for city, nv in VERIF.items():
        c = sorted(s[s.bolge == city].index)
        for i in rv.sample(c, nv):
            s.loc[i, "dogrulama"] = 1
    cols = ["audit_id", "split", "event", "tile", "bolge", "dur", "N_tabaka", "n_tabaka", "weight",
            "S1_rol", "S2_rol", "dogrulama", "n_intact", "n_damaged", "n_destroyed", "n_labeled"]
    s = s.sort_values("audit_id")[cols]
    out = Path(cfg["_root"]) / "data_manifest/denetim/denetim_orneklem_v1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} zaten var; önceden kayıtlı örneklem değiştirilmez.")
    s.to_csv(out, index=False)
    print("yazıldı:", out, len(s), "karo; sha1", hashlib.sha1(out.read_bytes()).hexdigest()[:10])
    print(s.groupby(["bolge", "dur"]).size().unstack(fill_value=0))
    print("doğrulama alt kümesi:", int(s.dogrulama.sum()))


if __name__ == "__main__":
    main()
