"""Manifest'te kullanılabilir karo sayısını doğrular (Adım 5'te Kaggle'da da aynı mantık).
    python -m src.check_manifest
"""
import sys
from pathlib import Path

import pandas as pd

from src.config import load


def main() -> int:
    cfg = load()
    m = pd.read_csv(Path(cfg["_root"]) / cfg["data"]["manifest"])
    n_ok = int((m["status"] == "ok").sum())
    exp = cfg["data"]["usable_tiles_expected"]
    print(f"satır: {len(m)}  status==ok: {n_ok}  beklenen: {exp}")
    print(m.loc[m["status"] == "ok"].groupby("event").size().rename("karo").to_string())
    return 0 if n_ok == exp else 1


if __name__ == "__main__":
    sys.exit(main())
