"""dm3_probs sentetik doğrulaması: bilinen bileşenler → beklenen sayılar. `python -m src.selftest_probs`"""
import numpy as np
from src.dm3_probs import ComponentDump
from src.dm3_eval import component_confusion, object_vote

H = W = 128
gt = np.zeros((H, W), np.uint8)
gt[10:30, 10:30] = 1          # sağlam bina (400 px)
gt[50:70, 50:70] = 3          # yıkık (400 px)
gt[90:100, 20:40] = 2         # hasarlı (200 px), model kaçıracak
valid = np.ones((H, W), bool)
loc = np.full((H, W), 0.05, np.float32)
loc[10:30, 10:30] = 0.9; loc[50:70, 50:70] = 0.8
loc[110:120, 100:120] = 0.9   # yanlış alarm (referansa uzak)
dam = np.zeros((4, H, W), np.float32); dam[0] = 1.0
dam[:, 10:30, 10:30] = np.array([0.1, 0.7, 0.15, 0.05], np.float32)[:, None, None]
dam[:, 50:70, 50:70] = np.array([0.1, 0.1, 0.2, 0.6], np.float32)[:, None, None]
dam[:, 110:120, 100:120] = np.array([0.1, 0.2, 0.3, 0.4], np.float32)[:, None, None]
pred = object_vote(loc, dam)
meta = dict(event=3, tile="e3_1", group="X", tier="T0", agir_bos=False)
d = ComponentDump(); d.add(meta, gt, valid, loc, dam, pred)
df = __import__("pandas").DataFrame(d.rows)
g = df[df.kind == "gt"].sort_values("cy"); f = df[df.kind == "fp"]
assert len(g) == 3 and len(f) == 1, (len(g), len(f))
assert list(g.gt_cls) == [1, 3, 2] and list(g.area) == [400, 400, 200]
assert abs(g.iloc[0].m1 - 0.7) < 1e-5 and abs(g.iloc[1].m3 - 0.6) < 1e-5
assert list(g.pred_cls)[2] == 0 and g.iloc[2].loc_cover == 0.0           # kaçırılan hasarlı
assert abs(f.iloc[0].min_dist_gt - 40) < 12 and f.iloc[0].pred_cls == 3
# dm3_eval ile tutarlılık: katı bileşen sayıları
C, fp, n_gt = component_confusion(gt, pred, valid, 0)
assert n_gt == 3 and int(fp.sum()) == len(f)
t = d.tiles[0]
assert t["t_n_pred_comp"] == 3 and t["t_loc_area"] == 400 + 400 + 200
print("selftest_probs: TAMAM")
