"""Veri yükleyici testleri sentetik karolarla: hizalama, 255 ignore, önbellek eşitliği, ChangeOS logit yolu.
    python -m src.synth_tiles --out /tmp/fake --per-group 3
    RAG_ROOT=/tmp/fake/repo RAG_DATA=/tmp/fake/data python -m src.selftest_dataset"""
import numpy as np
import torch
from pathlib import Path

from src.config import load
from src.dm3_dataset import EvalTiles, TrainTiles, collate_eval, collate_train, load_split, select_tiles, IGNORE
from src.dm3_verify import find_root


def main():
    cfg = load(); root = find_root()
    df = load_split("S1", cfg)
    tr, ev = select_tiles(df, holdout="Malatya")
    assert set(tr.group) == {"Adiyaman", "Osmaniye"} and set(ev.group) == {"Malatya"}
    for bad in (dict(holdout="Osmaniye"), dict(eval_role="test")):
        try:
            select_tiles(df, **bad); raise AssertionError("kilit çalışmadı")
        except SystemExit:
            pass
    s1 = select_tiles(df, holdout="Osmaniye", trigger_fold=True)
    assert set(s1[1].group) == {"Osmaniye"}
    assert "Gaziantep" not in set(tr.group)
    print("bölme/kilitler OK")

    ds = TrainTiles(tr, root, crop=256, crops_per_tile=4, dam_p=1.0, post_shift_px=0, photo=False, seed=1)
    x, d = ds[0]
    assert x.shape == (4, 6, 256, 256) and d.shape == (4, 256, 256) and d.dtype == torch.uint8
    # hizalama: öncesi parlaklığı bina maskesi içinde > dışında (sentetik binalar 150-230, zemin ~110 → normalize)
    for k in range(4):
        pre_gray = x[k, :3].mean(0); m = (d[k] > 0) & (d[k] != IGNORE)
        if m.sum() > 20 and (~m & (d[k] != IGNORE)).sum() > 20:
            assert pre_gray[m].mean() > pre_gray[(d[k] == 0)].mean() + 0.5, "öncesi–etiket hizası bozuk"
    assert (d >= 2).any(), "dam_p=1 hasarlı piksel içermiyor"
    print("kırpım hizası + hasar odaklı örnekleme OK")

    # ignore: boş öncesi şeridi olan karoda 255 görünmeli (sentetik karoların ~%20'sinde)
    ds2 = TrainTiles(tr, root, crop=1024, crops_per_tile=1, dam_p=0.0, photo=False, seed=2)
    n255 = sum(int((ds2[i][1] == IGNORE).any()) for i in range(len(ds2)))
    print(f"255 (geçersiz) içeren karo: {n255}/{len(ds2)}")
    assert n255 >= 1

    # önbellek eşitliği
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = TrainTiles(tr, root, crop=256, crops_per_tile=2, dam_p=0.5, post_shift_px=8, seed=3)
        b = TrainTiles(tr, root, crop=256, crops_per_tile=2, dam_p=0.5, post_shift_px=8, seed=3, cache_dir=td)
        for i in range(min(3, len(a))):
            xa, da = a[i]; xb, db = b[i]
            assert torch.equal(xa, xb) and torch.equal(da, db)
        xa2, _ = b[0]; assert torch.equal(xa2, a[0][0])     # ikinci okuma (önbellekten)
    print("önbellek eşitliği OK")

    # öteleme: yalnız sonrası değişir
    c0 = TrainTiles(tr, root, crop=256, crops_per_tile=1, dam_p=0.0, post_shift_px=0, photo=False, seed=5)
    c1 = TrainTiles(tr, root, crop=256, crops_per_tile=1, dam_p=0.0, post_shift_px=16, photo=False, seed=5)
    x0, d0 = c0[0]; x1, d1 = c1[0]
    assert torch.equal(x0[:, :3], x1[:, :3]) and torch.equal(d0, d1) and not torch.equal(x0[:, 3:], x1[:, 3:])
    print("öteleme yalnız sonrası görüntüde OK")

    evd = EvalTiles(ev, root)
    xe, ge, ve, me = collate_eval([evd[0]])
    assert xe.shape == (1, 6, 1024, 1024) and ge.max() <= 3 and ve.dtype == torch.bool and me[0]["group"] == "Malatya"
    print("eval karosu OK")

    # ChangeOS: bizim logit yolu == model.forward (eval) çıktıları
    from src.train_changeos import build_model, logits_fn
    m = build_model("r50", pretrained=False).eval()
    xt = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        ll, dl = logits_fn(m, xt); o = m(xt)
    assert torch.allclose(ll.sigmoid(), o.t1_semantic_prediction, atol=1e-5)
    assert torch.allclose(dl.softmax(1), o.change_prediction, atol=1e-5)
    print("logit yolu == ChangeOS.forward OK")
    print("TÜMÜ GEÇTİ")


if __name__ == "__main__":
    main()
