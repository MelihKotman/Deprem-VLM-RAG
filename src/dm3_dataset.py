"""Adım 8 / kutu 3 — DisasterM3 Türkiye veri yükleyici (ChangeOS için).

Bir karo = öncesi.png + sonrası.png (1024x1024 RGB) + üç bina maskesi (sağlam/hasarlı/yıkık; ikili PNG, >0 = o sınıf).
Eksik maske dosyası (manifest `mask_dosya` i/id/idd) = o sınıftan bina yok.

Etiket kodlaması (tek hasar haritası):  0 arka plan, 1 sağlam, 2 hasarlı, 3 yıkık   (çakışmada ağır hasar kazanır).
Konum etiketi = hasar > 0. Etiketler ÖNCESİ görüntüye hizalıdır; sonrası görüntü kaymış olabilir (manifest kayma_post_px).
Geçerlilik: src/tile_validity.valid_mask (öncesi/sonrası boş bölge). Eğitimde geçersiz piksel hasar hedefi 255 (ignore).

Eğitim modu (TrainTiles): karo başına K rastgele kırpım; kırpımların `dam_p` oranı hasarlı/yıkık piksel çevresinden seçilir;
  flip + rot90 (hepsine ortak); SONRASI görüntü kırpımı ±post_shift_px öteleme (yalnız sonrası; etiket öncesine bağlı kalır);
  hafif bağımsız parlaklık/kontrast oynatma (öncesi ve sonrası ayrı). Test/eval modunda hiçbir artırım yok (EvalTiles).

Bölme: split CSV'sinden (v1, dondurulmuş). Rol/grup seçimi ve test-kilidi için bkz. select_tiles().
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import load
from src.tile_validity import valid_mask

cv2.setNumThreads(1)
CLS = ("intact", "damaged", "destroyed")
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
IGNORE = 255


# ---------------------------------------------------------------- bölme seçimi
def tile_paths(root: Path, split: str, ev: int, tile: str):
    suf = tile.split("_", 2)[2]
    if split == "train":
        b, idr, pf = root / "train", "train_images", "train"
    else:
        b, idr, pf = root / "bench", "test_images", "test"
    pre = b / idr / f"turkey_earthquake{ev}_pre_{suf}.png"
    post = b / idr / f"turkey_earthquake{ev}_post_{suf}.png"
    masks = {c: b / "masks" / f"{pf}_building_{c}_mask" / f"{tile}.png" for c in CLS}
    return pre, post, masks


def load_split(scenario: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    df = pd.read_csv(Path(cfg["_root"]) / cfg["split"][scenario])
    man = pd.read_csv(Path(cfg["_root"]) / cfg["data"]["manifest"], usecols=["event", "tile", "dx_post", "dy_post"])
    df = df.merge(man, on=["event", "tile"], how="left")      # sonrası kayması (ETİKET kenarından ölçülmüş; yalnız 'oracle' hizalama tanısında kullanılır)
    val = pd.read_csv(Path(cfg["_root"]) / cfg["split"]["validity"])[["event", "tile", "gecerli_oran", "agir_bos"]]
    df = df.merge(val, on=["event", "tile"], how="left", validate="one_to_one")
    assert df.gecerli_oran.notna().all(), "geçerlilik tablosunda eksik karo"
    return df


def select_tiles(df: pd.DataFrame, *, holdout: str | None = None, trigger_fold: bool = False,
                 eval_role: str | None = None, allow_test: bool = False):
    """-> (train_df, eval_df | None).   Ön-kayıt korumaları:
      - test rolü yalnız allow_test=True ile açılır (S1: Antakya+e6+e7, S2: Osmaniye+e7);
      - Osmaniye katlaması (tetikleyici ölçümü) yalnız trigger_fold=True ile;
      - kontrol (Gaziantep) hiçbir zaman eğitimde.
    holdout: eğitim şehirlerinden biri dışarıda bırakılıp değerlendirilir (ayarlama katlaması).
    """
    tr = df[df.role == "train"]
    ev = None
    if holdout:
        if holdout not in set(tr.group):
            raise SystemExit(f"holdout '{holdout}' eğitim gruplarında yok: {sorted(set(tr.group))}")
        if holdout.lower().startswith("osmaniye") and not trigger_fold:
            raise SystemExit("Osmaniye katlaması tetikleyici ölçümüne ayrılmıştır (docs/adim8_tetikleyici.md): --trigger-fold gerekir")
        ev = tr[tr.group == holdout]
        tr = tr[tr.group != holdout]
    if eval_role == "test":
        if not allow_test:
            raise SystemExit("test bölmesi kilitli: yalnız nihai değerlendirmede --allow-test ile açılır")
        if holdout:
            raise SystemExit("--holdout ile test birlikte kullanılmaz")
        ev = df[df.role == "test"]
    elif eval_role == "control":
        ev = df[df.role == "control"]
    return tr.reset_index(drop=True), (None if ev is None else ev.reset_index(drop=True))


# ---------------------------------------------------------------- karo okuma
def read_tile(root: Path, split: str, ev: int, tile: str, shift=None):
    """-> pre RGB uint8 HxWx3, post RGB uint8, dam uint8 (0..3), valid bool.
    shift=(dx,dy): TANI (oracle hizalama) — sonrası, out(y,x)=post(y+dy,x+dx) olacak şekilde kaydırılır (manifest dx_post/dy_post
    sözleşmesi: score(G,E,dx,dy) = G[ys+dy, xs+dx]); kayan boş kenar geçersiz sayılır. Bu kayma ETİKETTEN ölçüldüğü için
    gerçek (etiketsiz) bir yöntem DEĞİLDİR; yalnız 'kayma sorunun nedeni mi' sorusunu yanıtlamak içindir."""
    pre_p, post_p, mp_ = tile_paths(root, split, ev, tile)
    pre = cv2.imread(str(pre_p), cv2.IMREAD_COLOR)
    post = cv2.imread(str(post_p), cv2.IMREAD_COLOR)
    if pre is None or post is None:
        raise FileNotFoundError(f"görüntü okunamadı: {pre_p} / {post_p}")
    if shift is not None and (shift[0] or shift[1]):
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        post = cv2.warpAffine(post, M, (post.shape[1], post.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid, _, _ = valid_mask(pre, post)
    dam = np.zeros(pre.shape[:2], np.uint8)
    for k, c in enumerate(CLS, start=1):
        if mp_[c].exists():
            m = cv2.imread(str(mp_[c]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                dam[m > 0] = k            # sıra: sağlam < hasarlı < yıkık (sonraki üstüne yazar)
    return cv2.cvtColor(pre, cv2.COLOR_BGR2RGB), cv2.cvtColor(post, cv2.COLOR_BGR2RGB), dam, valid


class _Base(Dataset):
    """Karoları (isteğe bağlı) disk önbelleğiyle okur. Önbellek: <cache_dir>/<tile>.npy, uint8 HxWx8
    (pre 3, post 3, dam 1, valid 1); mmap ile açılır — Kaggle /kaggle/working (20 GB) içinde ~8 MB/karo."""

    def __init__(self, df: pd.DataFrame, root: Path, cache_dir: str | None = None, align: bool = False):
        self.align = align
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _get(self, i):
        r = self.df.iloc[i]
        if self.cache:
            f = self.cache / f"{r.tile}{'_al' if self.align else ''}.npy"
            if not f.exists():
                pre, post, dam, valid = read_tile(self.root, r.split, int(r.event), r.tile, self._shift(r))
                arr = np.concatenate([pre, post, dam[..., None], valid.astype(np.uint8)[..., None]], axis=2)
                tmp = f.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp, arr)
                os.replace(tmp, f)
            a = np.load(f, mmap_mode="r")
            return r, a[..., 0:3], a[..., 3:6], a[..., 6], a[..., 7].astype(bool)
        pre, post, dam, valid = read_tile(self.root, r.split, int(r.event), r.tile, self._shift(r))
        return r, pre, post, dam, valid

    def _shift(self, r):
        if not self.align or not (pd.notna(r.dx_post) and pd.notna(r.dy_post)):
            return None
        return int(r.dx_post), int(r.dy_post)


def _norm(img_u8: np.ndarray) -> torch.Tensor:
    x = (img_u8.astype(np.float32) - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class TrainTiles(_Base):
    def __init__(self, df, root, *, crop=512, crops_per_tile=4, dam_p=0.5, post_shift_px=0, photo=True,
                 cache_dir=None, seed=42, align=False):
        super().__init__(df, root, cache_dir, align)
        self.crop, self.K, self.dam_p, self.R, self.photo = crop, crops_per_tile, dam_p, int(post_shift_px), photo
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        rng_shift = np.random.default_rng([self.seed, self.epoch, i, 1])   # öteleme ayrı akış: diğer artırımları değiştirmez
        r, pre, post, dam, valid = self._get(i)
        H, W = dam.shape
        S = self.crop
        dpix = None
        out_x, out_d = [], []
        for _ in range(self.K):
            if rng.random() < self.dam_p:
                if dpix is None:
                    dpix = np.argwhere(np.asarray(dam) >= 2)[::7]      # seyreltilmiş hasarlı/yıkık piksel listesi
                if len(dpix):
                    cy, cx = dpix[rng.integers(len(dpix))]
                    y0 = int(np.clip(cy - rng.integers(S // 8, S - S // 8), 0, H - S))
                    x0 = int(np.clip(cx - rng.integers(S // 8, S - S // 8), 0, W - S))
                else:
                    y0, x0 = int(rng.integers(0, H - S + 1)), int(rng.integers(0, W - S + 1))
            else:
                y0, x0 = int(rng.integers(0, H - S + 1)), int(rng.integers(0, W - S + 1))
            a = np.asarray(pre[y0:y0 + S, x0:x0 + S])
            if self.R > 0:
                dy, dx = (int(v) for v in rng_shift.integers(-self.R, self.R + 1, size=2))
            else:
                dy = dx = 0
            ys = int(np.clip(y0 + dy, 0, H - S)); xs = int(np.clip(x0 + dx, 0, W - S))
            b = np.asarray(post[ys:ys + S, xs:xs + S])
            d = np.asarray(dam[y0:y0 + S, x0:x0 + S]).copy()
            v = np.asarray(valid[y0:y0 + S, x0:x0 + S])
            d[~v] = IGNORE
            if rng.random() < 0.5:
                a, b, d = a[:, ::-1], b[:, ::-1], d[:, ::-1]
            if rng.random() < 0.5:
                a, b, d = a[::-1], b[::-1], d[::-1]
            k = int(rng.integers(4))
            if k:
                a, b, d = np.rot90(a, k), np.rot90(b, k), np.rot90(d, k)
            if self.photo:
                a = _jitter(a, rng); b = _jitter(b, rng)
            out_x.append(torch.cat([_norm(a), _norm(b)], 0))
            out_d.append(torch.from_numpy(np.ascontiguousarray(d)))
        return torch.stack(out_x), torch.stack(out_d)       # [K,6,S,S], [K,S,S] uint8 (0..3, 255)


def _jitter(img, rng, amt=0.10):
    c = 1.0 + rng.uniform(-amt, amt)
    b = rng.uniform(-amt, amt) * 255.0
    return np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)


def collate_train(batch):
    return torch.cat([b[0] for b in batch], 0), torch.cat([b[1] for b in batch], 0)


class EvalTiles(_Base):
    """Tam karo (1024x1024), artırımsız. -> x [6,H,W], dam [H,W] uint8, valid [H,W] bool, meta dict."""

    def __getitem__(self, i):
        r, pre, post, dam, valid = self._get(i)
        x = torch.cat([_norm(np.asarray(pre)), _norm(np.asarray(post))], 0)
        meta = dict(tile=r.tile, event=int(r.event), group=r.group, tier=str(r.kayma_tier),
                    agir_bos=int(r.agir_bos), shift=float(r.kayma_post_px) if pd.notna(r.kayma_post_px) else float("nan"))
        return x, torch.from_numpy(np.asarray(dam).copy()), torch.from_numpy(np.asarray(valid).copy()), meta


def collate_eval(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), [b[3] for b in batch])
