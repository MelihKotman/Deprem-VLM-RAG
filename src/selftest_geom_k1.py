"""Sentetik doğrulama: gölge yönü geri bulunur; çökmüş (gölgesiz) binalar yüksek tabakalı AUC; ters yön kontrolü ~0,5."""
import numpy as np, cv2, pandas as pd
from src.geom_k1 import tile_rows, choose_dir, city_stats, NDIR

rng = np.random.default_rng(1)
K_TRUE = 5                                  # 112,5 derece
ux, uy = np.cos(np.deg2rad(K_TRUE * 22.5)), np.sin(np.deg2rad(K_TRUE * 22.5))
rows = []
for tile in range(12):
    H = W = 640
    img = np.full((H, W, 3), 150, np.uint8); mi = np.zeros((H, W), np.uint8); md = np.zeros((H, W), np.uint8)
    placed = []
    for i in range(14):
        for _ in range(200):
            w, h = int(rng.integers(20, 44)), int(rng.integers(20, 44))
            x, y = int(rng.integers(60, W - 130)), int(rng.integers(60, H - 130))
            if all(abs(x - a) > 110 or abs(y - b) > 110 for a, b in placed): break
        else:
            continue
        placed.append((x, y)); collapsed = (i % 4 == 3)
        if not collapsed:
            for t in range(0, 26):                         # gölge: taban izinin t·u ötelenmişlerinin birleşimi
                sx, sy = int(round(t * ux)), int(round(t * uy))
                cv2.rectangle(img, (x + sx, y + sy), (x + sx + w - 1, y + sy + h - 1), (60, 60, 60), -1)
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (200, 200, 200), -1); mi[y:y + h, x:x + w] = 255
        else:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (170, 170, 170), -1); md[y:y + h, x:x + w] = 255
    img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    for r in tile_rows(img, {"intact": mi, "damaged": None, "destroyed": md}, np.ones((H, W), bool)):
        r["tile"] = tile; rows.append(r)
df = pd.DataFrame(rows); print("bina", len(df), df.cls.value_counts().to_dict())
k, sharp, _ = choose_dir(df); print("seçilen yön", k, "keskinlik %.3f" % sharp)
assert min((k - K_TRUE) % NDIR, (K_TRUE - k) % NDIR) <= 1
st = city_stats(df, nboot=100)
print("tabakalı AUC %.3f  ters-yön kontrol %.3f" % (st["tabakali_auc"], st["kontrol_ters_yon_auc"]))
assert st["tabakali_auc"] > 0.9 and abs(st["kontrol_ters_yon_auc"] - 0.5) < 0.2
print("SELFTEST K1: OK")
