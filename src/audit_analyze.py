"""Adım 6: denetim sayımından eşik kararı. Protokol: notlar/denetim-sonucu.md (sayımdan ÖNCE yazıldı).

Girdi (data_manifest/denetim/):
  denetim_orneklem_v1.csv            (audit_sample.py)
  sayfa_ozet.csv                     (audit_pages.py; pre_blank_frac)
  inceleme_ilk_gecis.csv             sütunlar: audit_id, n_eksik_gorunur, n_eksik_kapsamdisi, kayma (0/1/2), n_sinif_hata, not
  inceleme_dogrulama.csv (opsiyonel) aynı sütunlar, bağımsız ikinci sayıcı (yalnız dogrulama==1 karolar)

Oran = sum(w*eksik) / sum(w*(etiketli + eksik)); w = N_tabaka/n_tabaka (tabaka = şehir x hasar durumu).
Güven aralığı: tabaka içinde karo yeniden örnekleme (bootstrap), 5000 tekrar, seed 42, yüzdelik %95.
Karar (yalnız S1 test tabakaları havuzlanmış: Antakya + e6 + e7):
  nokta <= 0.10 ve üst sınır <= 0.10 -> "<= %10: etiketler olduğu gibi"
  nokta <= 0.10 ve üst sınır  > 0.10 -> "belirsiz: denetim genişletilir (Melih'e danış)"
  nokta  > 0.10                      -> "> %10: Adım 7a eksik bina tamamlama"
    python -m src.audit_analyze [--dir data_manifest/denetim] [--sayfa-ozet <yol>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load

THR = 0.10
B, SEED = 5000, 42


def ratio(df, w=True):
    ww = df.weight if w else 1.0
    num = (ww * df.miss).sum(); den = (ww * (df.n_labeled + df.miss)).sum()
    return num / den if den else np.nan


def boot(df, w=True):
    rng = np.random.default_rng(SEED)
    groups = [g.reset_index(drop=True) for _, g in df.groupby(["bolge", "dur"])]
    vals = []
    for _ in range(B):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        vals.append(ratio(pd.concat(parts), w))
    return np.nanpercentile(vals, [2.5, 97.5])


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b)); po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def table(d):
    groups = {
        "Antakya/Defne": d.bolge == "Antakya/Defne", "e6": d.bolge == "bilinmiyor(event6)",
        "e7": d.bolge == "bilinmiyor(event7)", "Adiyaman": d.bolge == "Adiyaman",
        "Malatya": d.bolge == "Malatya", "Osmaniye": d.bolge == "Osmaniye", "Gaziantep": d.bolge == "Gaziantep",
        "S1 TEST (Antakya+e6+e7)": d.S1_rol == "test", "S2 test (Osmaniye+e7)": d.S2_rol == "test",
        "S1 egitim (Adiyaman+Malatya+Osmaniye)": d.S1_rol == "egitim",
    }
    rows = []
    for name, msk in groups.items():
        g = d[msk]
        lo, hi = boot(g, True)
        rows.append(dict(grup=name, karo=len(g), etiketli=int(g.n_labeled.sum()), eksik=int(g.miss.sum()),
                         oran_agirlikli=ratio(g, True), ci_lo=lo, ci_hi=hi, oran_agirliksiz=ratio(g, False),
                         kapsamdisi_eksik=int(g.n_eksik_kapsamdisi.sum()),
                         kayma_2_karo=int((g.kayma == 2).sum()), sinif_hata=int(g.n_sinif_hata.sum()),
                         ort_pre_bos=round(float(g.pre_blank_frac.mean()), 3)))
    return pd.DataFrame(rows)


def decide(t):
    r = t[t.grup.str.startswith("S1 TEST")].iloc[0]
    if r.oran_agirlikli > THR:
        s = f"> %10 (nokta {r.oran_agirlikli:.1%}, %95 AA {r.ci_lo:.1%}-{r.ci_hi:.1%}) -> Adım 7a: test bölmesinde eksik bina tamamlama"
    elif r.ci_hi > THR:
        s = f"BELİRSİZ (nokta {r.oran_agirlikli:.1%} <= %10 ama AA üst sınırı {r.ci_hi:.1%} > %10) -> denetim genişletilir, Melih'e danış"
    else:
        s = f"<= %10 (nokta {r.oran_agirlikli:.1%}, %95 AA {r.ci_lo:.1%}-{r.ci_hi:.1%}) -> etiketler olduğu gibi kullanılır"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_manifest/denetim")
    ap.add_argument("--sayfa-ozet")
    a = ap.parse_args(argv)
    D = Path(load()["_root"]) / a.dir
    s = pd.read_csv(D / "denetim_orneklem_v1.csv")
    so = pd.read_csv(a.sayfa_ozet or D / "sayfa_ozet.csv")[["audit_id", "pre_blank_frac"]]
    r1 = pd.read_csv(D / "inceleme_ilk_gecis.csv")
    assert set(r1.audit_id) == set(s.audit_id), "ilk geçiş tablosu 96 karonun tamamını içermeli"
    d = s.merge(so, on="audit_id").merge(r1.drop(columns=["not"], errors="ignore"), on="audit_id")
    d["miss"] = d.n_eksik_gorunur
    t = table(d)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.round(4).to_string(index=False))
    t.to_csv(D / "denetim_ozet.csv", index=False)
    print("\nKARAR:", decide(t))
    vp = D / "inceleme_dogrulama.csv"
    if vp.exists():
        v = pd.read_csv(vp).drop(columns=["not"], errors="ignore")
        m = d.merge(v, on="audit_id", suffixes=("", "_m"))
        print(f"\nDoğrulama alt kümesi: {len(m)} karo")
        print("  eksik sayısı: ortalama |fark| =", round((m.miss - m.n_eksik_gorunur_m).abs().mean(), 2),
              "; Spearman =", round(m.miss.corr(m.n_eksik_gorunur_m, method="spearman"), 3))
        m2 = m.copy(); m2["miss"] = m2.n_eksik_gorunur_m
        print("  oran (ilk geçiş sayımı, aynı alt küme):", round(ratio(m, False), 4),
              "| (Melih sayımı):", round(ratio(m2, False), 4))
        print("  kayma derecesi: tam uyum =", round((m.kayma == m.kayma_m).mean(), 3),
              "; kappa =", round(kappa(m.kayma, m.kayma_m), 3))
        print("  sınıf hatası sayısı: ortalama |fark| =", round((m.n_sinif_hata - m.n_sinif_hata_m).abs().mean(), 2))
        d2 = d.copy()
        mm = m.set_index("audit_id").n_eksik_gorunur_m
        d2["miss"] = [mm.get(i, x) for i, x in zip(d2.audit_id, d2.miss)]
        t2 = table(d2)
        print("  duyarlılık (alt kümede Melih sayımı kullanılırsa) kararı:", decide(t2))


if __name__ == "__main__":
    main()
