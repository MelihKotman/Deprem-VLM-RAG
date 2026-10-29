"""dm3_eval + dm3_dataset birim testleri (model/GPU/gerçek veri gerekmez). Çalıştır: python -m src.selftest_eval"""
import numpy as np
from src.dm3_eval import Acc, object_vote, trigger_r, zone

META = dict(tile="t", event=1, group="G", tier="T0", agir_bos=0)


def scene(S=256, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.zeros((S, S), np.uint8)
    boxes = [(20, 20, 40, 40, 1), (100, 30, 40, 40, 2), (30, 120, 40, 40, 3), (150, 150, 40, 40, 3), (200, 40, 30, 30, 1)]
    for x, y, w, h, c in boxes:
        gt[y:y + h, x:x + w] = c
    return gt


def run(pred, gt, valid=None):
    valid = np.ones_like(gt, bool) if valid is None else valid
    a = Acc(); a.update([("ALL",)], gt, pred, valid, META)
    return a.summary()["ALL"]


def shift(m, dx, dy):
    return np.roll(np.roll(m, dx, 1), dy, 0)


def main():
    gt = scene()
    # 1. kusursuz tahmin
    r = run(gt.copy(), gt)
    for lv in ("strict", "tol5", "tol15"):
        assert r[lv]["f1_intact"] == r[lv]["f1_damaged"] == r[lv]["f1_destroyed"] == 1.0, (lv, r[lv])
        assert r[lv]["n_gt_components"] == 5 and r[lv]["n_missed"] == 0 and sum(r[lv]["n_false_alarm"]) == 0
    assert r["pixel"]["miou_3"] == 1.0 and r["pixel"]["loc_iou"] == 1.0
    print("1 kusursuz tahmin OK")
    # 2. 25 px kayık tahmin: katı bozulur, d=15 yetmez (25>15) ama d=30 benzeri yok → tol15 < 1; 12 px kayıkta tol15 = 1
    p12 = shift(gt, 12, 0)
    r12 = run(p12, gt)
    assert r12["strict"]["damage_f1"] <= r12["tol5"]["damage_f1"] <= r12["tol15"]["damage_f1"], r12
    assert r12["tol15"]["f1_destroyed"] == 1.0 and r12["tol15"]["f1_damaged"] == 1.0, r12["tol15"]
    print("2 kayma: katı %.2f ≤ tol5 %.2f ≤ tol15 %.2f OK" % (r12["strict"]["damage_f1"], r12["tol5"]["damage_f1"], r12["tol15"]["damage_f1"]))
    # 2b. küçük (12 px) binalar + 25 px kayma (boşluk 13 px): katı hepsini kaçırır, tol15 kurtarır, tol5 kurtarmaz
    g2 = np.zeros((256, 256), np.uint8)
    for k, (x, y, c) in enumerate([(20, 20, 2), (100, 30, 3), (30, 120, 3), (150, 150, 2), (200, 40, 3)]):
        g2[y:y + 12, x:x + 12] = c
    r2 = run(shift(g2, 25, 0), g2)
    assert r2["strict"]["n_missed"] == 5 and r2["strict"]["damage_f1"] == 0.0, r2["strict"]
    assert r2["tol5"]["n_missed"] == 5, r2["tol5"]
    assert r2["tol15"]["damage_f1"] == 1.0 and r2["tol15"]["n_missed"] == 0, r2["tol15"]
    assert trigger_r(r2) == 0.0 and zone(trigger_r(r2)) == "karo_es_ana"
    print("2b küçük bina kayması: katı 0 / tol5 0 / tol15 1 → r=0 OK")
    # 3. sınıf hatası: yıkık → hasarlı
    p = gt.copy(); p[p == 3] = 2
    r = run(p, gt)
    assert r["strict"]["f1_destroyed"] == 0.0 and r["strict"]["f1_damaged"] < 1.0
    assert r["strict"]["confusion_true_x_pred0123"][2][2] == 2
    print("3 sınıf hatası OK")
    # 4. kaçırılan + yanlış alarm
    p = gt.copy(); p[gt == 2] = 0                     # hasarlı bina kaçırıldı
    p[230:250, 230:250] = 3                           # örtüşmeyen yıkık tahmin
    r = run(p, gt)
    assert r["strict"]["n_missed"] == 1 and r["strict"]["n_false_alarm"] == [0, 0, 1], r["strict"]
    assert r["tol15"]["n_missed"] == 1 and r["tol15"]["n_false_alarm"] == [0, 0, 1]
    print("4 kaçırılan/FP OK")
    # 5. geçerlilik maskesi: yıkık bina tamamen geçersizse sayılmaz
    valid = np.ones_like(gt, bool); valid[120:170, 0:80] = False      # (30,120,40,40) binası dışarıda
    r = run(gt.copy(), gt, valid)
    assert r["strict"]["n_gt_components"] == 4 and r["strict"]["f1_destroyed"] == 1.0
    print("5 geçerlilik maskesi OK")
    # 6. nesne oylaması: bileşen içi ortalama olasılık
    S = 64
    loc = np.zeros((S, S), np.float32); loc[10:30, 10:30] = 0.9; loc[40:60, 40:60] = 0.9
    dam = np.zeros((4, S, S), np.float32); dam[0] = 0.1
    dam[1] = 0.2; dam[2] = 0.1; dam[3] = 0.1
    dam[3, 10:30, 10:25] = 0.8; dam[1, 10:30, 10:25] = 0.05      # bileşen 1: 15/20 sütun yıkık → yıkık
    dam[2, 40:60, 40:60] = 0.7                                   # bileşen 2: hasarlı
    v = object_vote(loc, dam)
    assert (v[10:30, 10:30] == 3).all() and (v[40:60, 40:60] == 2).all() and v[0, 0] == 0
    print("6 nesne oylaması OK")
    # 7. tetikleyici
    r = run(shift(gt, 12, 0), gt)
    rr = trigger_r(r)
    assert rr is not None and 0 <= rr <= 1.0 and zone(0.6) == "karo_es_ana" and zone(0.8) == "ara_bolge" and zone(0.9) == "yalniz_bina"
    assert trigger_r({"strict": {"damage_f1": 0.0}, "tol15": {"damage_f1": 0.0}}) is None
    print("7 tetikleyici r = %.2f (%s) OK" % (rr, zone(rr)))
    # 8. oylama ağırlığı: ağırlıksızda sağlam kazanan bileşen, ağırlıkla yıkığa döner
    from src.dm3_eval import vote_weights
    loc = np.zeros((32, 32), np.float32); loc[4:20, 4:20] = 0.9
    dam = np.zeros((4, 32, 32), np.float32); dam[1] = 0.6; dam[3] = 0.3
    assert (object_vote(loc, dam)[4:20, 4:20] == 1).all()
    w = vote_weights([3000, 100, 100], "inv")
    assert w == (1.0, 30.0, 30.0) and (object_vote(loc, dam, weights=w)[4:20, 4:20] == 3).all()
    assert vote_weights([3000, 100, 100], "none") is None and abs(vote_weights([3000, 100, 100], "sqrt_inv")[1] - 30 ** 0.5) < 1e-9
    print("8 oylama ağırlığı OK")
    print("TÜMÜ GEÇTİ")


if __name__ == "__main__":
    main()
