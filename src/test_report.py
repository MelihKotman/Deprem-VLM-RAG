"""Ön-kayıtlı rapor (docs/adim8_tam_egitim_onkayit.md §6): per_tile.csv -> sınıf bazlı F1, hasar-F1, Macro-F1 ve ikili "hasar var/yok" F1,
karo bazlı bootstrap %95 aralığı (B=2000, seed 42; yüzdelik). Katı / tol5 / tol15 yan yana; kırılımlar: ALL, group, tier.

  python -m src.test_report --runs s1_seed42=/yol/per_tile.csv s1_seed43=... --out /yol/rapor [--levels 0,5,15]

Tanımlar (src/dm3_eval.py ile aynı): F1 sınıf başına bina bileşeni üzerinden; hasar-F1 = (hasarlı F1 + yıkık F1)/2; Macro-F1 = 3 sınıf ortalaması.
İkili "hasar var/yok": gerçek hasarlı|yıkık bileşen ↔ tahmin hasarlı|yıkık; TP = gerçek{2,3} tahmin{2,3}; FN = gerçek{2,3} - TP;
FP = (gerçek sağlam, tahmin {2,3}) + yanlış alarm bileşenleri sınıf {2,3}. Pilot hold-out bulgusundan doğdu; test'ten bağımsız kaydedildi.
Karo bazlı bootstrap: karolar bağımsız örnek sayılır (event içi örtüşme Adım 14'te ayrıca ele alınır; daha kaba küme yalnız aralığı genişletir).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_NAME = {0: "strict", 5: "tol5", 15: "tol15"}
B = 2000
SEED = 42
METRICS = ("f1_intact", "f1_damaged", "f1_destroyed", "damage_f1", "macro_f1", "binary_damage_f1")


def _cols(t):
    return [f"C{t}_t{r}p{c}" for r in (1, 2, 3) for c in range(4)], [f"fp{t}_c{c}" for c in (1, 2, 3)]


def tile_counts(df, t):
    cc, fc = _cols(t)
    miss = [c for c in cc + fc if c not in df.columns]
    if miss:
        raise SystemExit(f"per_tile.csv'de karo sayaçları yok ({miss[0]}...): eski sürümle üretilmiş; değerlendirmeyi güncel kodla yeniden yap")
    return np.concatenate([df[cc].to_numpy(np.float64), df[fc].to_numpy(np.float64)], axis=1)   # [n, 15]


def metrics_from_sums(S):
    """S [..., 15] (12 karışım + 3 yanlış alarm toplamı) -> dict(metrik -> [...])"""
    C = S[..., :12].reshape(S.shape[:-1] + (3, 4)); fp = S[..., 12:]
    out = {}
    f = {}
    for c in (1, 2, 3):
        tp = C[..., c - 1, c]; fn = C[..., c - 1, :].sum(-1) - tp; fpc = C[..., :, c].sum(-1) - tp + fp[..., c - 1]
        den = 2 * tp + fn + fpc
        with np.errstate(divide="ignore", invalid="ignore"):
            f[c] = np.where(den > 0, 2 * tp / den, np.nan)
    out["f1_intact"], out["f1_damaged"], out["f1_destroyed"] = f[1], f[2], f[3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["damage_f1"] = np.nanmean(np.stack([f[2], f[3]]), axis=0)
        out["macro_f1"] = np.nanmean(np.stack([f[1], f[2], f[3]]), axis=0)
    tp = C[..., 1:3, 2:4].sum((-2, -1)); gt = C[..., 1:3, :].sum((-2, -1)); fn = gt - tp
    fpb = C[..., 0, 2:4].sum(-1) + fp[..., 1] + fp[..., 2]
    den = 2 * tp + fn + fpb
    with np.errstate(divide="ignore", invalid="ignore"):
        out["binary_damage_f1"] = np.where(den > 0, 2 * tp / den, np.nan)
    return out


def bootstrap(X, rng):
    """X [n,15] -> (nokta, lo, hi) her metrik için"""
    pt = {k: float(v) for k, v in metrics_from_sums(X.sum(0)).items()}
    n = len(X)
    idx = rng.integers(0, n, size=(B, n))
    boots = metrics_from_sums(X[idx].sum(1))
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in METRICS:
            v = boots[k]
            ok = v[~np.isnan(v)]
            res[k] = (pt[k], float(np.percentile(ok, 2.5)) if len(ok) > B // 2 else np.nan,
                      float(np.percentile(ok, 97.5)) if len(ok) > B // 2 else np.nan)
    return res


def run_report(name, df, levels=(0, 5, 15)):
    rows = []
    subsets = [("ALL", "ALL", df)]
    subsets += [("group", g, d) for g, d in df.groupby("group")]
    subsets += [("tier", t, d) for t, d in df.groupby("tier")]
    for lv in levels:
        Xall = tile_counts(df, lv)
        for kind, key, d in subsets:
            X = Xall[d.index.to_numpy()]
            rng = np.random.default_rng(SEED)          # her alt küme/ düzey için aynı tohum: yeniden üretilebilir
            res = bootstrap(X, rng)
            gt_by = X[:, :12].reshape(len(X), 3, 4).sum((0, 2)).astype(int)
            row = dict(run=name, level=LEVEL_NAME[lv], subset=kind, key=key, n_tiles=len(d),
                       n_gt_intact=gt_by[0], n_gt_damaged=gt_by[1], n_gt_destroyed=gt_by[2])
            for k in METRICS:
                p, lo, hi = res[k]
                row[k], row[k + "_lo"], row[k + "_hi"] = p, lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(r, k):
    v, lo, hi = r[k], r[k + "_lo"], r[k + "_hi"]
    if v != v:
        return "—"
    return f"{v:.3f} [{'—' if lo != lo else format(lo, '.3f')}, {'—' if hi != hi else format(hi, '.3f')}]"


def to_markdown(R):
    lines = []
    for lv in ("strict", "tol15", "tol5"):
        for run in R.run.unique():
            sub = R[(R.run == run) & (R.level == lv)]
            lines += [f"### {run} — {lv}", "",
                      "| kırılım | karo | bina (sağlam/hasarlı/yıkık) | F1 sağlam | F1 hasarlı | F1 yıkık | hasar-F1 | Macro-F1 | ikili hasar var/yok F1 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                lines.append(f"| {r['subset']}:{r['key']} | {r.n_tiles} | {r.n_gt_intact}/{r.n_gt_damaged}/{r.n_gt_destroyed} | "
                             + " | ".join(fmt(r, k) for k in METRICS) + " |")
            lines.append("")
    return "\n".join(lines)


def seed_summary(R):
    """aynı kırılım ve düzeyde koşular arası ortalama ± std (nokta tahminleri üzerinden)"""
    g = R.groupby(["level", "subset", "key"])
    out = g[list(METRICS)].agg(["mean", "std", "min", "max"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="ad=per_tile.csv yolu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0,5,15")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(x) for x in a.levels.split(","))
    parts = []
    for spec in a.runs:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path).reset_index(drop=True)
        parts.append(run_report(name, df, levels))
    R = pd.concat(parts, ignore_index=True)
    R.to_csv(out / "rapor_uzun.csv", index=False)
    seed_summary(R).to_csv(out / "rapor_seed_ozeti.csv", index=False)
    (out / "rapor.md").write_text(to_markdown(R))
    print(f"yazıldı: {out}/rapor.md, rapor_uzun.csv, rapor_seed_ozeti.csv ({len(R)} satır)")
    print(R[(R.subset == "ALL")][["run", "level", "n_tiles", "damage_f1", "macro_f1", "binary_damage_f1"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
