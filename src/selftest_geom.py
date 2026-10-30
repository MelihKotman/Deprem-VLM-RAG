"""Sentetik doğrulama: bilinen ofsetli dikdörtgen binalar -> building_offsets ofseti ±STEP içinde geri bulur; AUC mekaniği çalışır."""
import numpy as np, cv2, pandas as pd
from src.geom_k0 import building_offsets, gradient_map, auc, city_stats

rng = np.random.default_rng(0)
H = W = 512
masks = np.zeros((H, W), np.uint8); post = np.full((H, W, 3), 60, np.uint8)
truth = []
for i in range(12):
    while True:
        x, y = int(rng.integers(40, W - 120)), int(rng.integers(40, H - 120)); w, h = int(rng.integers(20, 40)), int(rng.integers(20, 40))
        if masks[max(0, y - 40):y + h + 40, max(0, x - 40):x + w + 40].sum() == 0: break
    dx, dy = (int(v) for v in rng.choice(np.arange(-12, 13, 2), 2))
    masks[y:y + h, x:x + w] = 255
    cv2.rectangle(post, (x + dx, y + dy), (x + dx + w - 1, y + dy + h - 1), (200, 200, 200), -1)
    truth.append((x, y, dx, dy))
valid = np.ones((H, W), bool)
rows = building_offsets(gradient_map(post), {"intact": masks, "damaged": None, "destroyed": None}, valid, np.array([1.0, 0.0]))
assert len(rows) == 12, len(rows)
got = sorted((r["dx"], r["dy"]) for r in rows); exp = sorted((dx, dy) for _, _, dx, dy in truth)
bad = sum(1 for a, b in zip(got, exp) if abs(a[0] - b[0]) > 2 or abs(a[1] - b[1]) > 2)
print("ofset sapması >2 px olan:", bad, "/ 12"); assert bad <= 1
# AUC mekaniği
assert abs(auc(np.array([0., 1, 2, 3]), np.array([0, 0, 1, 1], bool)) - 1.0) < 1e-9
assert abs(auc(np.array([1., 1, 1, 1]), np.array([0, 0, 1, 1], bool)) - 0.5) < 1e-9
d = pd.DataFrame(dict(tile=np.repeat(np.arange(10), 20), proj=np.r_[rng.normal(0, 3, 100), rng.normal(0, 3, 100)] , area=rng.integers(40, 900, 200),
                      skor_tepe=rng.random(200), pos=np.tile([False] * 10 + [True] * 10, 10)))
st = city_stats(d, nboot=50); print("gürültü AUC:", round(st["auc"], 3)); assert 0.3 < st["auc"] < 0.7
# sinyal: pozitifler -4 px daha az yaslanır -> skor -proj yüksek -> AUC > 0.7
d.loc[d.pos, "proj"] -= 4; st = city_stats(d, nboot=50); print("sinyal AUC:", round(st["auc"], 3)); assert st["auc"] > 0.7
print("SELFTEST GEOM: OK")
