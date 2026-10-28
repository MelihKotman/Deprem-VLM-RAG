"""config/base.yaml yükleyici. Kullanım:
    python -m src.config            # yapılandırmayı ve çözülen yolları yazdırır
    from src.config import load     # cfg = load(); cfg["seed"]
Proje kökü: RAG_ROOT ortam değişkeni, yoksa bu dosyanın iki üstü.
"""
import hashlib
import os
from pathlib import Path

import yaml


def root() -> Path:
    env = os.environ.get("RAG_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[1]


def load(path: str = "config/base.yaml") -> dict:
    p = root() / path
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root())
    cfg["_config_sha1"] = hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    return cfg


def path(cfg: dict, key: str) -> Path:
    """paths altındaki anahtarı mutlak yola çevirir."""
    return Path(cfg["_root"]) / cfg["paths"][key]


if __name__ == "__main__":
    cfg = load()
    print("kök       :", cfg["_root"])
    print("config sha:", cfg["_config_sha1"])
    print("seed      :", cfg["seed"], cfg["seeds"])
    for k in ("data_manifest", "notebooks", "kb", "runs", "reports"):
        p = path(cfg, k)
        print(f"{k:14s}:", p, "VAR" if p.exists() else "YOK")
    m = Path(cfg["_root"]) / cfg["data"]["manifest"]
    print("manifest  :", m.name, "VAR" if m.exists() else "YOK")
