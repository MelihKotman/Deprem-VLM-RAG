"""test_report birim testi (model/GPU gerekmez): per_tile sayaçları, Acc.summary() ile aynı F1'i vermeli. python -m src.selftest_report"""
import numpy as np
import pandas as pd
from src.dm3_eval import Acc
from src.selftest_eval import scene
from src.test_report import metrics_from_sums, tile_counts, run_report, bootstrap


def main():
    rng = np.random.default_rng(0)
    acc = Acc()
    for i in range(12):
        gt = scene(seed=i)
        pred = np.roll(gt, int(rng.integers(-14, 15)), axis=1).copy()
        flip = rng.random(pred.shape) < 0.0
        pred[(pred == 2) & (rng.random() < 0.5)] = 3          # bazı karolarda hasarlı -> yıkık
        meta = dict(tile=f"t{i}", event=1, group="G1" if i < 6 else "G2", tier="T0" if i % 2 else "T1", agir_bos=0)
        valid = np.ones_like(gt, bool)
        acc.update([("ALL",), ("group", meta["group"])], gt, pred, valid, meta)
    df = pd.DataFrame(acc.tile_rows)
    S = acc.summary()["ALL"]
    for lv, name in ((0, "strict"), (5, "tol5"), (15, "tol15")):
        m = metrics_from_sums(tile_counts(df, lv).sum(0))
        for k_rep, k_acc in (("f1_intact", "f1_intact"), ("f1_damaged", "f1_damaged"), ("f1_destroyed", "f1_destroyed"),
                             ("damage_f1", "damage_f1"), ("macro_f1", "macro_f1")):
            a, b = float(m[k_rep]), S[name][k_acc]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (lv, k_rep, a, b)
        # ikili F1'i karışım matrisinden elle hesapla
        C = np.array(S[name]["confusion_true_x_pred0123"]); fp = np.array(S[name]["n_false_alarm"])
        tp = C[1:3, 2:4].sum(); fn = C[1:3].sum() - tp; fpb = C[0, 2:4].sum() + fp[1] + fp[2]
        want = 2 * tp / (2 * tp + fn + fpb) if (2 * tp + fn + fpb) else float("nan")
        assert abs(float(m["binary_damage_f1"]) - want) < 1e-9, (lv, m["binary_damage_f1"], want)
    print("1 per_tile sayaçları Acc.summary() ile aynı F1 + ikili F1 OK")
    # 2 bootstrap: aralık nokta tahmini kapsar, yeniden üretilebilir
    X = tile_counts(df, 15)
    r1 = bootstrap(X, np.random.default_rng(42)); r2 = bootstrap(X, np.random.default_rng(42))
    assert r1 == r2
    for k, (p, lo, hi) in r1.items():
        if p == p and lo == lo:
            assert lo - 1e-9 <= p <= hi + 1e-9, (k, p, lo, hi)
    print("2 bootstrap tekrarlanabilir, aralık nokta tahminini kapsıyor OK")
    # 3 rapor tablosu kırılımları
    R = run_report("t", df)
    assert set(R.subset) == {"ALL", "group", "tier"} and set(R.level) == {"strict", "tol5", "tol15"}
    assert R[(R.subset == "ALL") & (R.level == "strict")].n_tiles.iloc[0] == 12
    print("3 rapor kırılımları OK")
    # 4 boş hasar (tüm hasar sınıfı yok) -> NaN, çökmez
    z = np.zeros((1, 15)); z[0, 0] = 5   # yalnız sağlam bileşenler doğru
    mm = metrics_from_sums(z.sum(0))
    assert np.isnan(mm["f1_damaged"]) and np.isnan(mm["binary_damage_f1"])
    print("4 boş sınıf -> NaN OK\nTÜMÜ GEÇTİ")


if __name__ == "__main__":
    main()
