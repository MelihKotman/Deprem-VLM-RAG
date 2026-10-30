"""Adım 8B-1 analizi — kalibrasyon, kararsız bant (Kapı T), F1-optimum eşik (ön-kayıt: docs/adim8b_onkayit.md §3–§5).
Girdi: şehir-bırakmalı katlamaların components.csv.gz (src/dm3_probs.py). Test bölmesine DOKUNMAZ.
Kullanım: python -m src.adim8b_kalibrasyon <ciktilar/adim-8b1 klasörü>

Tanımlar (sonuçlara bakmadan bu dosyada sabit; birincil olanlar işaretli):
  birim            : referans bina bileşeni (kind=gt). y = gt_cls in {2,3}. p = (m2+m3)/(m1+m2+m3) (ham, oylamasız).
  Kapı T (BİRİNCİL): havuzlanmış dışarıdaki şehirlerde, gerçek hasarlı∪yıkık binaların p∈[0,3;0,7] bandındaki payı ≥ %20.
                     (tüm referans binalar; ikincil: yalnız yakalananlar loc_cover≥0,5)
  ECE              : 10 eşit genişlikli kutu + 10 nicel kutu (ikisi de raporlanır; olasılıklar çok küçük olduğundan eşit genişlik tek başına yanıltıcı).
  (i) çıplak argmax: pred_cls∈{2,3}; yanlış alarm = sağlam bina pred≥2 + fp satırları pred≥2 (katı, tampon 0).
  (ii) kalibre+eşik: diğer iki şehirden öğrenilen izotonik kalibrasyon (yalnız gt satırları) + F1'i maksimize eden eşik (gt+fp satırları); üçüncü şehre uygulanır.
  Karo bazlı bootstrap (2000) eşleştirilmiş ΔF1 = (ii) − (i).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

CITIES = {"Adiyaman": "fold_adiyaman_seed42", "Malatya": "fold_malatya_seed42", "Osmaniye": "fold_osmaniye_seed42_eval"}
rng = np.random.default_rng(42)


def load(root):
    out = {}
    for c, d in CITIES.items():
        df = pd.read_csv(Path(root) / d / "components.csv.gz")
        den = (df.m1 + df.m2 + df.m3).clip(lower=1e-12)
        df["p"] = (df.m2 + df.m3) / den
        df["y"] = (df.gt_cls >= 2).astype(int)
        df["pred_pos"] = (df.pred_cls >= 2).astype(int)
        df["city"] = c
        out[c] = df
    return out


def auc(y, p):
    y = np.asarray(y); r = pd.Series(p).rank(method="average").values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def ap(y, p):
    o = np.argsort(-np.asarray(p), kind="stable"); y = np.asarray(y)[o]
    tp = np.cumsum(y); prec = tp / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def ece(y, p, nb=10, mode="width"):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if mode == "width":
        b = np.minimum((p * nb).astype(int), nb - 1)
    else:
        b = np.minimum((pd.Series(p).rank(method="first").values - 1) * nb // len(p), nb - 1).astype(int)
    e = 0.0
    for k in range(nb):
        m = b == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


class Iso:
    def fit(self, p, y):
        o = np.argsort(p, kind="stable"); p = np.asarray(p)[o]; y = np.asarray(y, float)[o]
        blocks = []   # [sum_y, n, max_p]
        for pi, yi in zip(p, y):
            blocks.append([yi, 1, pi])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][2] = b[2]
        self.hi = np.array([b[2] for b in blocks]); self.val = np.array([b[0] / b[1] for b in blocks])
        return self

    def __call__(self, p):
        i = np.minimum(np.searchsorted(self.hi, p, side="left"), len(self.val) - 1)
        return self.val[i]


def f1_counts(df, score, thr):
    pos = score >= thr
    tp = int(((df.y == 1) & pos).sum()); fp = int(((df.y == 0) & pos).sum())
    fn = int(((df.y == 1) & ~pos).sum())
    return tp, fp, fn


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def best_thr(df, score):
    cand = np.unique(np.quantile(score, np.linspace(0, 1, 2001)))
    best = (-1, None)
    for t in cand:
        if t <= 0: continue
        v = f1(*f1_counts(df, score, t))
        if v == v and v > best[0]: best = (v, float(t))
    return best


def tile_boot(df, pos_a, pos_b, n=2000):
    """eşleştirilmiş karo bootstrap: ΔF1 = F1(b) − F1(a)."""
    key = df.event.astype(str) + "_" + df.tile.astype(str)
    tiles = key.unique(); idx = {t: i for i, t in enumerate(tiles)}; ti = key.map(idx).values
    def cnt(pos):
        a = np.zeros((len(tiles), 3))
        np.add.at(a[:, 0], ti, ((df.y == 1) & pos).values); np.add.at(a[:, 1], ti, ((df.y == 0) & pos).values)
        np.add.at(a[:, 2], ti, ((df.y == 1) & ~pos).values); return a
    A, B = cnt(pos_a), cnt(pos_b); d = []
    for _ in range(n):
        s = rng.integers(0, len(tiles), len(tiles))
        a, b = A[s].sum(0), B[s].sum(0)
        d.append(f1(*b) - f1(*a))
    d = np.array(d); return float(np.nanmean(d)), [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def main(root):
    root = Path(root); D = load(root); res = {"per_city": {}, "pooled": {}, "loco": {}}
    # --- doğrulama: dökümün baz çizgisi, dm3_eval karışım matrisiyle uyuşuyor mu (katı)
    for c, d in CITIES.items():
        mp = root / d / "metrics.json"
        if not mp.exists(): continue
        s = json.load(open(mp))["by"]["ALL"]["strict"]; C = np.array(s["confusion_true_x_pred0123"])
        tp = C[1:, 2:].sum(); fn = C[1:, :2].sum(); fpg = C[0, 2:].sum(); fpc = sum(s["n_false_alarm"][1:])
        g = D[c]; pr = g.pred_pos == 1
        mine = (int(((g.y == 1) & pr).sum()), int(((g.y == 0) & pr).sum()), int(((g.y == 1) & ~pr).sum()))
        res["per_city"].setdefault(c, {})["check_vs_metrics"] = dict(metrics=[int(tp), int(fpg + fpc), int(fn)], dump=list(mine),
                                                                   ok=bool(mine == (int(tp), int(fpg + fpc), int(fn))))
    allg = pd.concat([d[d.kind == "gt"] for d in D.values()])
    for name, g in list(((c, D[c][D[c].kind == "gt"]) for c in D)) + [("POOLED", allg)]:
        pos, neg = g[g.y == 1], g[g.y == 0]
        r = dict(n=len(g), n_pos=len(pos), prev=len(pos) / len(g), auc=auc(g.y, g.p), ap=ap(g.y, g.p),
                 ece_width=ece(g.y, g.p), ece_quant=ece(g.y, g.p, mode="quant"),
                 gate_T_share_pos_in_band=float(((pos.p >= .3) & (pos.p <= .7)).mean()),
                 gate_T_share_pos_detected_in_band=float(((pos[pos.loc_cover >= .5].p >= .3) & (pos[pos.loc_cover >= .5].p <= .7)).mean()) if (pos.loc_cover >= .5).any() else float("nan"),
                 share_neg_in_band=float(((neg.p >= .3) & (neg.p <= .7)).mean()), n_pos_detected=int((pos.loc_cover >= .5).sum()),
                 p_quantiles_pos={q: float(pos.p.quantile(q)) for q in (.1, .25, .5, .75, .9, .99)},
                 p_quantiles_neg={q: float(neg.p.quantile(q)) for q in (.1, .25, .5, .75, .9, .99)},
                 p_max=float(g.p.max()), share_p_gt_0p1_pos=float((pos.p > .1).mean()), share_p_gt_0p1_neg=float((neg.p > .1).mean()))
        res["per_city"].setdefault(name, {}).update(r) if name != "POOLED" else res["pooled"].update(r)
    # --- LOCO (ii) vs (i)
    for X in D:
        others = [c for c in D if c != X]
        tr = pd.concat([D[c] for c in others]); te = D[X]
        trg = tr[tr.kind == "gt"]; iso = Iso().fit(trg.p.values, trg.y.values)
        tr_s = iso(tr.p.values); te_s = iso(te.p.values)
        fb, thr = best_thr(tr.assign(), tr_s)
        base = f1(*f1_counts(te, te.pred_pos.values.astype(float), 1))
        tp, fp, fn = f1_counts(te, te_s, thr)
        pos_a = (te.pred_pos == 1); pos_b = pd.Series(te_s >= thr, index=te.index)
        dm, ci = tile_boot(te.reset_index(drop=True), pos_a.reset_index(drop=True), pos_b.reset_index(drop=True))
        tp0, fp0, fn0 = f1_counts(te, te.pred_pos.values.astype(float), 1)
        res["loco"][X] = dict(train_cities=others, thr_calibrated=thr, train_best_f1=fb,
                              f1_baseline=base, prec_base=tp0 / max(tp0 + fp0, 1), rec_base=tp0 / max(tp0 + fn0, 1),
                              f1_ii=f1(tp, fp, fn), prec_ii=tp / max(tp + fp, 1), rec_ii=tp / max(tp + fn, 1),
                              delta_f1=dm, delta_ci95=ci, n_fp_rows_in_test=int((te.kind == "fp").sum()),
                              lipton_check=dict(F_star_train=fb, F_star_over_2=fb / 2, thr=thr),
                              fp_ii=fp, fp_base=fp0, tp_ii=tp, tp_base=tp0, n_pos=int(te.y.sum()))
    out = root / "analiz"; out.mkdir(exist_ok=True)
    (out / "kalibrasyon_sonuc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        for j, c in enumerate(D):
            g = D[c][D[c].kind == "gt"]
            bins = np.logspace(-6, 0, 40)
            ax[0, j].hist(np.clip(g[g.y == 0].p, 1e-6, 1), bins=bins, alpha=.6, label=f"sağlam n={int((g.y==0).sum())}", density=True)
            ax[0, j].hist(np.clip(g[g.y == 1].p, 1e-6, 1), bins=bins, alpha=.6, label=f"hasarlı∪yıkık n={int(g.y.sum())}", density=True)
            ax[0, j].set_xscale("log"); ax[0, j].axvspan(.3, .7, color="gray", alpha=.2); ax[0, j].set_title(f"{c}: p_hasar dağılımı"); ax[0, j].legend()
            q = pd.qcut(g.p.rank(method="first"), 10, labels=False)
            m = g.groupby(q).agg(p=("p", "mean"), y=("y", "mean"))
            ax[1, j].plot(m.p, m.y, "o-"); mx = max(m.p.max(), m.y.max()) * 1.1
            ax[1, j].plot([0, mx], [0, mx], "k--", lw=.8); ax[1, j].set_title(f"{c}: güvenilirlik (10 nicel kutu)"); ax[1, j].set_xlabel("ortalama p"); ax[1, j].set_ylabel("gerçek hasar oranı")
        plt.tight_layout(); plt.savefig(out / "kalibrasyon.png", dpi=110)
    except Exception as e:
        print("şekil atlandı:", e)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
