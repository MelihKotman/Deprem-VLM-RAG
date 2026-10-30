"""Adım 8B-1 — bina bileşeni başına sınıf olasılıklarının dökümü (kalibrasyon / kararsız bant / ikinci aşama girdisi).

ÖN-KAYIT: docs/adim8b_onkayit.md §3 (8B-1). Yeni mimari YOK; yalnız mevcut modelin çıktısı kaydedilir. Oylama/ölçüt kodu (dm3_eval) DEĞİŞMEDİ.
Yalnız DIŞARIDAKİ şehir (veya kontrol) üzerinde anlamlıdır; eğitim-içi olasılıklar iyimserdir.

İki satır türü (components.csv.gz, sütun `kind`):
  gt : referans bina bileşeni (katı, tampon 0; geçerlilik maskesi uygulanmış, 8-komşuluk — dm3_eval.component_confusion ile aynı tanım).
       Bu bileşen İÇİNDEKİ piksellerin ham (oylamasız, kalibre edilmemiş) ortalamaları: m0..m3 (softmax hasar kanalları), loc_mean (sigmoid konum),
       loc_cover (konum>0.5 piksel payı). gt_cls = referans çoğunluk sınıfı (1 sağlam, 2 hasarlı, 3 yıkık); g1..g3 = referans sınıf piksel sayıları
       (karışık bileşen payı için). pred_cls = oylamalı katı tahmin (0 = kaçırılan). Yakalanamayan bina da satırdadır (loc_cover ≈ 0 olabilir).
  fp : tahmin bileşeni (konum>0.5, 8-komşuluk) ve referans bina pikseliyle KATI örtüşmesi YOK. min_dist_gt = en yakın referans bina pikseline px uzaklığı
       (tol5 / tol15 yanlış alarmı = min_dist_gt > 5 / > 15). Referansla örtüşen tahmin bileşenleri ayrı satır OLMAZ (gt satırının içindedir).
Ortak sütunlar: event, tile, group, tier, agir_bos, comp (karo içi sıra), area (px), cx, cy (merkez, px).
Sıkıştırma ile ikili hasar olasılığı sonradan türetilir: p_dmg = (m2+m3)/(m1+m2+m3). Önsel (oylama ağırlığı) kaydırması burada UYGULANMAZ.

tile_scores.csv: karo skorları (konum>0.5 piksellerde): t_pdmg_mean, t_pdmg_max (tahmin bileşeni ortalamalarının en büyüğü), t_p3_max, t_loc_area, t_n_pred_comp.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def _comp_stats(lab, n, loc, dam, xs, ys):
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    a = np.maximum(area, 1.0)
    loc_mean = np.bincount(flat, weights=loc.ravel(), minlength=n) / a
    loc_cover = np.bincount(flat, weights=(loc.ravel() > 0.5).astype(np.float64), minlength=n) / a
    m = np.stack([np.bincount(flat, weights=dam[c].ravel(), minlength=n) / a for c in range(4)], 1)
    cx = np.bincount(flat, weights=xs, minlength=n) / a
    cy = np.bincount(flat, weights=ys, minlength=n) / a
    return area, loc_mean, loc_cover, m, cx, cy


class ComponentDump:
    def __init__(self):
        self.rows = []
        self.tiles = []
        self._grid = None

    def _xy(self, shape):
        if self._grid is None or self._grid[0] != shape:
            h, w = shape
            self._grid = (shape, np.tile(np.arange(w, dtype=np.float64), h), np.repeat(np.arange(h, dtype=np.float64), w))
        return self._grid[1], self._grid[2]

    def add(self, tile, gt, valid, loc, dam, pred):
        """tile: meta sözlüğü (tile, event, group, tier, agir_bos); gt uint8 [H,W] 0..3; valid [H,W]; loc [H,W] sigmoid; dam [4,H,W] softmax; pred oylamalı uint8."""
        v = np.asarray(valid).astype(bool)
        g = np.where(v, gt, 0).astype(np.uint8)
        p = np.where(v, pred, 0).astype(np.uint8)
        loc = np.asarray(loc, np.float32); dam = np.asarray(dam, np.float32)
        xs, ys = self._xy(g.shape)
        base = dict(event=tile["event"], tile=tile["tile"], group=tile["group"], tier=tile["tier"], agir_bos=tile["agir_bos"])
        gloc = (g > 0).astype(np.uint8)
        # ---- referans bileşenleri
        if gloc.any():
            n, lab = cv2.connectedComponents(gloc, connectivity=8)
            area, lm, lc, m, cx, cy = _comp_stats(lab, n, loc, dam, xs, ys)
            gcnt = np.bincount((lab.astype(np.int64) * 4 + g).ravel(), minlength=n * 4).reshape(n, 4)
            pcnt = np.bincount((lab.astype(np.int64) * 4 + p).ravel(), minlength=n * 4).reshape(n, 4)
            for k in range(1, n):
                if gcnt[k, 1:].sum() == 0:
                    continue
                pc = int(pcnt[k, 1:].argmax() + 1) if pcnt[k, 1:].sum() > 0 else 0
                self.rows.append(dict(base, kind="gt", comp=k, area=int(area[k]), cx=round(cx[k], 1), cy=round(cy[k], 1),
                                      gt_cls=int(gcnt[k, 1:].argmax() + 1), g1=int(gcnt[k, 1]), g2=int(gcnt[k, 2]), g3=int(gcnt[k, 3]),
                                      pred_cls=pc, loc_mean=float(lm[k]), loc_cover=float(lc[k]),
                                      m0=float(m[k, 0]), m1=float(m[k, 1]), m2=float(m[k, 2]), m3=float(m[k, 3]), min_dist_gt=0.0))
        # ---- tahmin bileşenleri (yanlış alarm adayları + karo skoru)
        ploc = (loc > 0.5).astype(np.uint8)
        npc, plab = cv2.connectedComponents(ploc, connectivity=8)
        t = dict(base, t_loc_area=int(ploc.sum()), t_n_pred_comp=int(npc - 1), t_pdmg_mean=0.0, t_pdmg_max=0.0, t_p3_max=0.0)
        if npc > 1:
            area, lm, lc, m, cx, cy = _comp_stats(plab, npc, loc, dam, xs, ys)
            den = np.maximum(m[:, 1] + m[:, 2] + m[:, 3], 1e-9)
            pdmg = (m[:, 2] + m[:, 3]) / den
            t["t_pdmg_max"] = float(pdmg[1:].max()); t["t_p3_max"] = float(m[1:, 3].max())
            inpred = ploc.ravel() > 0
            dpix = dam[2].ravel()[inpred] + dam[3].ravel()[inpred]
            ppix = dam[1].ravel()[inpred] + dpix
            t["t_pdmg_mean"] = float(np.mean(dpix / np.maximum(ppix, 1e-9)))
            if gloc.any():
                dist = cv2.distanceTransform((gloc == 0).astype(np.uint8), cv2.DIST_L2, 5)
            else:
                dist = np.full(g.shape, 1e4, np.float32)
            sel = plab.ravel() > 0                                   # yalnız tahmin pikselleri (scipy'siz en küçük uzaklık)
            mind = np.full(npc, np.inf); np.minimum.at(mind, plab.ravel()[sel], dist.ravel()[sel])
            mind = mind[1:]
            pcnt = np.bincount((plab.astype(np.int64) * 4 + p).ravel(), minlength=npc * 4).reshape(npc, 4)
            for k in range(1, npc):
                if mind[k - 1] > 0:
                    pc = int(pcnt[k, 1:].argmax() + 1) if pcnt[k, 1:].sum() > 0 else 0
                    self.rows.append(dict(base, kind="fp", comp=k, area=int(area[k]), cx=round(cx[k], 1), cy=round(cy[k], 1),
                                          gt_cls=0, g1=0, g2=0, g3=0, pred_cls=pc, loc_mean=float(lm[k]), loc_cover=float(lc[k]),
                                          m0=float(m[k, 0]), m1=float(m[k, 1]), m2=float(m[k, 2]), m3=float(m[k, 3]),
                                          min_dist_gt=float(mind[k - 1])))
        self.tiles.append(t)

    def save(self, out):
        from pathlib import Path
        out = Path(out)
        pd.DataFrame(self.rows).to_csv(out / "components.csv.gz", index=False, compression="gzip")
        pd.DataFrame(self.tiles).to_csv(out / "tile_scores.csv", index=False)
        print(f"olasılık dökümü -> {out / 'components.csv.gz'} ({len(self.rows)} satır), tile_scores.csv ({len(self.tiles)} karo)")
