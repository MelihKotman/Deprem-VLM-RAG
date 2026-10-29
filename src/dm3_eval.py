"""Adım 8 / kutu 3 — 3 sınıflı nesne oylaması + değerlendirme (docs/adim7_olcut_tanimlari.md, docs/adim8_tetikleyici.md).

ORTAK SON İŞLEM (tüm görsel modellere aynı): konum olasılığı > 0.5 → bağlı bileşenler (8-komşuluk) → her bileşen için
hasar olasılıklarının (sınıf 1..3) ortalaması → en büyük olan sınıf. Ağırlık YOK (ChangeOS'un xBD'ye sabit 8/38/25/11 oylaması kullanılmaz).
Arka plan kanalı (0) oylamaya girmez.

Ölçüt düzeyleri (hepsi geçerlilik maskesi içinde; tamamı dışarıda kalan bina bileşeni sayılmaz):
  katı bina düzeyi   : referans bina bileşeni (bağlı bileşen) içinde tahmin edilen piksellerin çoğunluk sınıfı (tampon 0 px)
  toleranslı (d px)  : aynı, ama tahmin, bileşenin d px çevresinde aranır; her piksel EN YAKIN referans bileşene atanır
                       (çakışan genişletmelerde çift sayım yok; docs'taki "genişletilmiş çevre"nin uygulaması — sapma notu)
  bileşende tahmin yok → "kaçırılan" (tahmin sınıfı 0); referansla örtüşmeyen tahmin bileşeni → yanlış alarm (FP)
  piksel düzeyi      : 4x4 karışım matrisi → sınıf IoU; konum IoU
  karo düzeyi (ikincil): "yıkık bina var mı" (karo başına), hasar yoğunluğu = (hasarlı+yıkık alan) / bina alanı (MAE, Spearman)
F1 sınıf başına bina bileşeni üzerinden; hasar-F1 = (hasarlı F1 + yıkık F1) / 2; Macro-F1 = 3 sınıf ortalaması.
Toplama: tüm karolar üzerinde sayaçlar toplanır (mikro), kırılımlar: grup(event/şehir), kayma kademesi, genel.
"""
from __future__ import annotations

import cv2
import numpy as np

NC = 4          # 0 arka plan + 3 hasar sınıfı
TOLS = (0, 5, 15)   # 0 = katı


def object_vote(loc_prob: np.ndarray, dam_prob: np.ndarray, thr: float = 0.5, weights=None) -> np.ndarray:
    """loc_prob [H,W] (sigmoid), dam_prob [4,H,W] (softmax) → pred hasar haritası uint8 (0..3).
    weights: sınıf 1..3 için çarpan (None = ağırlıksız). Eğitim kümesi bina sayılarından türetilir (vote_weights), test görülmeden."""
    loc = (loc_prob > thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(loc, connectivity=8)
    if n <= 1:
        return np.zeros(loc.shape, np.uint8)
    flat = lab.ravel()
    sums = np.stack([np.bincount(flat, weights=dam_prob[c].ravel(), minlength=n) for c in (1, 2, 3)], 1)  # [n,3]
    if weights is not None:
        sums = sums * np.asarray(weights, np.float64)[None, :]
    cls = (sums.argmax(1) + 1).astype(np.uint8)
    cls[0] = 0
    return cls[lab]


def _comp_counts(lab, dam, n):
    """lab [H,W] int, dam [H,W] uint8 → [n,4] piksel sayıları (bileşen x sınıf)."""
    return np.bincount((lab.astype(np.int64) * NC + dam).ravel(), minlength=n * NC).reshape(n, NC)


def component_confusion(gt, pred, valid, tol):
    """-> (C [3,4], fp [3], n_gt).  C[t-1, p]: gerçek sınıf t (1..3), tahmin p (0=kaçırılan,1..3)."""
    g = np.where(valid, gt, 0).astype(np.uint8)
    p = np.where(valid, pred, 0).astype(np.uint8)
    gloc = (g > 0).astype(np.uint8)
    C = np.zeros((3, NC), np.int64); fp = np.zeros(3, np.int64)
    if gloc.sum() == 0:
        n_gt = 0
        region = np.zeros_like(gloc, bool); lab = np.zeros(gloc.shape, np.int32)
    else:
        if tol <= 0:
            n, lab = cv2.connectedComponents(gloc, connectivity=8)
            region = gloc.astype(bool)
        else:
            inv = (gloc == 0).astype(np.uint8)          # sıfırlar = referans bina pikselleri
            dist, lab = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
            region = dist <= tol
            n = int(lab.max()) + 1
            # CCOMP etiketleri yalnız sıfır pikseller bileşenlerine ait; arka plan pikselleri en yakın bileşenin etiketini alır
        n = int(lab.max()) + 1
        gcnt = _comp_counts(np.where(gloc > 0, lab, 0), g, n)             # referans sınıfı: yalnız bina pikselleri
        pcnt = _comp_counts(np.where(region, lab, 0), np.where(region, p, 0), n)
        ids = np.arange(1, n)
        gclass = gcnt[1:, 1:].argmax(1) + 1                               # [n-1] 1..3
        has_g = gcnt[1:, 1:].sum(1) > 0
        pv = pcnt[1:, 1:]
        pclass = np.where(pv.sum(1) > 0, pv.argmax(1) + 1, 0)
        for t, q in zip(gclass[has_g], pclass[has_g]):
            C[t - 1, q] += 1
        n_gt = int(has_g.sum())
    # yanlış alarm: referans bölgesiyle (tol ile genişletilmiş) örtüşmeyen tahmin bileşenleri
    ploc = (p > 0).astype(np.uint8)
    npc, plab = cv2.connectedComponents(ploc, connectivity=8)
    if npc > 1:
        overl = np.bincount(plab[region & (ploc > 0)].ravel(), minlength=npc) > 0
        pc = _comp_counts(plab, p, npc)[:, 1:]
        for k in range(1, npc):
            if not overl[k] and pc[k].sum() > 0:
                fp[pc[k].argmax()] += 1
    return C, fp, n_gt


class Acc:
    """Sayaç toplayıcı: anahtar → sayaçlar. Kırılımlar update(...) çağrısındaki keys listesiyle."""

    def __init__(self):
        self.d = {}
        self.tile_rows = []

    def _slot(self, key):
        if key not in self.d:
            self.d[key] = dict(
                comp={t: [np.zeros((3, NC), np.int64), np.zeros(3, np.int64), 0] for t in TOLS},
                pix=np.zeros((NC, NC), np.int64),
                n_tiles=0,
                dstr_gt=[], dstr_pr=[],       # karo düzeyi yoğunluk
                destroyed_present=np.zeros((2, 2), np.int64),   # [gt var, pred var]
            )
        return self.d[key]

    def update(self, keys, gt, pred, valid, tile):
        comp = {t: component_confusion(gt, pred, valid, t) for t in TOLS}
        v = valid.astype(bool)
        pix = np.bincount((gt[v].astype(np.int64) * NC + pred[v]), minlength=NC * NC).reshape(NC, NC)
        g_b = ((gt > 0) & v).sum(); p_b = ((pred > 0) & v).sum()
        dg = float(((gt >= 2) & v).sum() / g_b) if g_b else float("nan")
        dp = float(((pred >= 2) & v).sum() / p_b) if p_b else float("nan")
        gd = int(comp[0][0][2].sum() > 0); pd_ = int(((pred == 3) & v).sum() > 0)
        # karoda yıkık bina var mı: gerçek = yıkık bileşen sayısı > 0 ; tahmin = yıkık tahmin bileşeni var
        self.tile_rows.append(dict(tile=tile["tile"], event=tile["event"], group=tile["group"], tier=tile["tier"],
                                   agir_bos=tile["agir_bos"], n_gt=comp[0][2], dens_gt=dg, dens_pred=dp,
                                   gt_destroyed=gd, pred_destroyed=pd_))
        for k in keys:
            s = self._slot(k)
            s["n_tiles"] += 1
            for t in TOLS:
                C, fp, n = comp[t]
                s["comp"][t][0] += C; s["comp"][t][1] += fp; s["comp"][t][2] += n
            s["pix"] += pix
            s["dstr_gt"].append(dg); s["dstr_pr"].append(dp)
            s["destroyed_present"][gd, pd_] += 1

    # ---------------------------------------------------------------- özet
    @staticmethod
    def _f1(C, fp):
        out = {}
        for c in (1, 2, 3):
            tp = C[c - 1, c]; fn = C[c - 1].sum() - tp; fpc = C[:, c].sum() - tp + fp[c - 1]
            den = 2 * tp + fn + fpc
            out[c] = float(2 * tp / den) if den else float("nan")
        return out

    def summary(self):
        res = {}
        for key, s in self.d.items():
            r = dict(n_tiles=s["n_tiles"])
            for t in TOLS:
                C, fp, n = s["comp"][t]
                f = self._f1(C, fp)
                name = "strict" if t == 0 else f"tol{t}"
                r[name] = dict(
                    f1_intact=f[1], f1_damaged=f[2], f1_destroyed=f[3],
                    macro_f1=float(np.nanmean(list(f.values()))) if not all(np.isnan(list(f.values()))) else float("nan"),
                    damage_f1=float(np.nanmean([f[2], f[3]])) if not (np.isnan(f[2]) and np.isnan(f[3])) else float("nan"),
                    n_gt_components=int(n), n_gt_by_class=[int(x) for x in C.sum(1)],
                    n_missed=int(C[:, 0].sum()), n_false_alarm=[int(x) for x in fp],
                    confusion_true_x_pred0123=C.tolist())
            P = s["pix"]
            ious = {}
            for c in (1, 2, 3):
                den = P[c].sum() + P[:, c].sum() - P[c, c]
                ious[c] = float(P[c, c] / den) if den else float("nan")
            locI = P[1:, 1:].sum(); locU = P[1:].sum() + P[:, 1:].sum() - locI
            r["pixel"] = dict(iou_intact=ious[1], iou_damaged=ious[2], iou_destroyed=ious[3],
                              miou_3=float(np.nanmean(list(ious.values()))),
                              loc_iou=float(locI / locU) if locU else float("nan"))
            dp = s["destroyed_present"]
            tp = dp[1, 1]; fn = dp[1, 0]; fpp = dp[0, 1]
            r["tile_level"] = dict(destroyed_present_f1=float(2 * tp / (2 * tp + fn + fpp)) if (2 * tp + fn + fpp) else float("nan"),
                                   destroyed_present_confusion_gt_x_pred=dp.tolist())
            gt_, pr_ = np.array(s["dstr_gt"]), np.array(s["dstr_pr"])
            pr_ok = np.where(np.isnan(pr_), 0.0, pr_)          # tahmin bina yok → yoğunluk 0
            ok = ~np.isnan(gt_)
            if ok.sum() > 2:
                from scipy.stats import spearmanr
                rho = spearmanr(gt_[ok], pr_ok[ok]).statistic
                r["tile_level"].update(density_mae=float(np.abs(gt_[ok] - pr_ok[ok]).mean()),
                                       density_spearman=float(rho) if rho == rho else float("nan"), n_density=int(ok.sum()))
            res[" | ".join(map(str, key)) if isinstance(key, tuple) else str(key)] = r
        return res


def trigger_r(summary_all: dict):
    """Tetikleyici: r = katı hasar-F1 / toleranslı(d=15) hasar-F1 (docs/adim8_tetikleyici.md). Payda 0 → None."""
    a = summary_all["strict"]["damage_f1"]; b = summary_all["tol15"]["damage_f1"]
    if b is None or b != b or b == 0:
        return None
    return float(a / b)


def zone(r):
    if r is None:
        return "tanimsiz"
    return "karo_es_ana" if r < 0.70 else ("ara_bolge" if r < 0.85 else "yalniz_bina")


def vote_weights(n_by_class, mode):
    """Eğitim kümesi bina sayılarından (sağlam, hasarlı, yıkık) oylama ağırlığı; sağlam = 1.
    none → None; inv → N_sağlam/N_c; sqrt_inv → sqrt(N_sağlam/N_c). Sınıf önsel düzeltmesi (ChangeOS'un sabit 8/38/25/11'inin veriye bağlı karşılığı)."""
    if mode == "none":
        return None
    n = np.maximum(np.asarray(n_by_class, np.float64), 1.0)
    w = n[0] / n
    return tuple((np.sqrt(w) if mode == "sqrt_inv" else w).tolist())
