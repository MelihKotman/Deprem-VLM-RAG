# Adım 8 · ChangeOS kurulabilirlik testi (Kaggle) — 28 Ekim 2026

Amaç: ChangeOS (torchange) Kaggle T4'te kurulup fp16 ile eğitim adımı atabiliyor mu? Gerçek veri yok, rastgele tensör. Çalışırsa ana model ChangeOS; çalışmazsa DamFormer-benzeri (kendi yazdığımız).

## Bulut ortamında (CPU) ön deneme sonucu
- `pip install ever-beta timm git+https://github.com/Z-Zheng/pytorch-change-models.git` → `ever-beta` 0.6.x ile **ResNet kodlayıcı bozuk** (`AttributeError: ... norm_layer`; 0.6.0'da gelen otomatik-config sarmalayıcı `set_default_config`'i boşa çıkarıyor).
- **Çözüm: `pip install "ever-beta==0.5.6"`** (torchange 0.0.5 metadata'sı `>=0.6.1` ister, pip uyarı verir; çalışmayı engellemez).
- Python 3.13 + torch 2.15 CPU: `r50` (39,05 M param) ve `swint` (41,45 M param) ileri-geri yayılım, 4 sınıflı hasar başı, kayıp sonlu → OK. CPU hızı anlamsız (≈2,4 s/adım, 256 px, batch 2).
- fp16/GPU BU ORTAMDA DENENMEDİ → Kaggle testinin asıl konusu.

## Kaggle'da çalıştırma (GPU T4 ×2, internet açık)
Hücre 1:
```
!pip install -q timm "ever-beta==0.5.6"
!pip install -q --no-deps "git+https://github.com/Z-Zheng/pytorch-change-models.git"
!pip install -q albumentations tifffile einops "datasets[vision]"
```
(`--no-deps` torch'u ezmesin diye.) Sonra repo kökünde:
```
!python -m src.changeos_check --encoder r50   --crop 512 --batch 4 --steps 20 --fp16 --out /kaggle/working/cos_r50_fp16.json
!python -m src.changeos_check --encoder swint --crop 512 --batch 4 --steps 20 --fp16 --out /kaggle/working/cos_swint_fp16.json
!python -m src.changeos_check --encoder r50   --crop 512 --batch 8 --steps 20 --fp16 --out /kaggle/working/cos_r50_b8.json
```
Başarı ölçütü: `"status": "OK"`, `loss_finite: true` (fp16'da NaN yok), `peak_mem_GB` < 15, `s_per_step` ve `img_per_s` kayıtlı.
Başarısızlıkta: hata metnini olduğu gibi yapıştır (kurulum mu, fp16 mi, CUDA mı ayrışsın).

## Bilinen noktalar (kod okumasından)
- `ChangeOS` kodunda fp16 yok; `torch.autocast` + `GradScaler` ile biz sarmalıyoruz (script öyle yapıyor).
- Kayıplar: konum BCE + Tversky(0,9/0,1); hasar CE + Dice (ignore=255).
- `_object_vote` xBD'ye sabit: sınıf listesi [1,2,3,4], ağırlık (8, 38, 25, 11). Bizde 3 hasar sınıfı → kendi oylamamızı yazacağız (aynısını tüm modellere ortak son işlem yapmak için).
- Girdi: `[B, 6, H, W]` (öncesi+sonrası kanal birleşik), `y = {masks: [konum, hasar]}`.
- BRIGHT/xBD checkpoint'i yüklenmez; yalnız ImageNet omurga (ResNet `pretrained=True`, Swin-T torchvision IMAGENET1K_V1).
