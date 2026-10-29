"""runs/*/ (metrics.json + timing.json + config_used.json) → tek tablo.   python -m src.summarize_runs /kaggle/working/runs"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def main(root):
    rows = []
    for d in sorted(Path(root).iterdir()):
        m, c = d / "metrics.json", d / "config_used.json"
        if not (m.exists() and c.exists()):
            continue
        M, C = json.loads(m.read_text()), json.loads(c.read_text())
        A = C["args"]; a = M["by"].get("ALL", {})
        t = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
        rows.append((d.name, A["scenario"], A.get("holdout") or A.get("eval_role") or "sanity", A["encoder"], A["epochs"],
                     A["post_shift"], A["dam_p"], A.get("cls_weights") or "-", A["seed"],
                     f(a.get("strict", {}).get("damage_f1")), f(a.get("tol15", {}).get("damage_f1")), f(a.get("tol5", {}).get("damage_f1")),
                     f(a.get("strict", {}).get("macro_f1")), f(a.get("pixel", {}).get("miou_3")), f(a.get("pixel", {}).get("loc_iou")),
                     f(t.get("mean_epoch_s"), 0), f(t.get("data_wait_frac"), 2), f(t.get("peak_mem_GB"), 1),
                     "SANITY" if M.get("eval_is_sanity_in_sample") else ""))
    hdr = ("koşu", "senaryo", "değerlendirme", "enc", "ep", "öteleme", "dam_p", "cls_w", "seed", "katı hasarF1", "tol15 hasarF1",
           "tol5 hasarF1", "katı macroF1", "piksel mIoU", "konum IoU", "ep sn", "veri bekl.", "bellek GB", "not")
    print(" | ".join(hdr)); print(" | ".join("---" for _ in hdr))
    for r in rows:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
