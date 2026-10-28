"""DisasterM3 Türkiye dosya düzeninden karo durumu çıkarır (yalnız dosya adları; görüntü açılmaz).

Mantık `data_manifest/.../manifest.csv`'yi üreten eski taramayla aynı:
  status == ok  <=>  öncesi + sonrası optik + en az bir bina maskesi var.
Girdi: split kökünden göreli POSIX yollar (örn. `masks/test_building_intact_mask/turkey_earthquake1_10_11.png`).
Hem diskteki klasörü (verify) hem uzak dosya listesini (download --dry-run) tarar.
"""
import re
from collections import defaultdict

CLS = ("intact", "damaged", "destroyed")
_MASK = re.compile(r"^masks/(train|test)_building_(intact|damaged|destroyed)_mask/turkey_earthquake(\d+)_(.+)\.png$")
_IMG = re.compile(r"^(train|test)_images/turkey_earthquake(\d+)_(pre|post|sar_post)_(.+)\.png$")


def tiles_from_paths(split: str, relpaths) -> dict:
    """-> {(split, event, tile): {'status': ok|maskesiz|optik_eksik, 'pre','post','sar','masks'}}"""
    pref = "train" if split == "train" else "test"
    T = defaultdict(lambda: {"pre": False, "post": False, "sar": False, "masks": set()})
    for p in relpaths:
        m = _MASK.match(p)
        if m and m.group(1) == pref:
            ev, suf = int(m.group(3)), m.group(4)
            T[(ev, suf)]["masks"].add(m.group(2))
            continue
        m = _IMG.match(p)
        if m and m.group(1) == pref:
            ev, kind, suf = int(m.group(2)), m.group(3), m.group(4)
            T[(ev, suf)]["sar" if kind == "sar_post" else kind] = True
    out = {}
    for (ev, suf), d in T.items():
        if not d["masks"]:
            st = "maskesiz"
        elif d["pre"] and d["post"]:
            st = "ok"
        else:
            st = "optik_eksik"
        out[(split, ev, f"turkey_earthquake{ev}_{suf}")] = {**d, "status": st}
    return out
