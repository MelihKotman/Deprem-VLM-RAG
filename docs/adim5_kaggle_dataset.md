# Adım 5 — Veriyi Kaggle'a taşı (kılavuz)

Hedef: DisasterM3 Türkiye (~5 GB) özel Kaggle dataset'i; bilgisayara hiçbir şey inmez.

## A. Yükleme (bir kez)
1. Kaggle → *Create → New Notebook*; ad: `00-kaggle-veri-yukle`. *File → Import Notebook* ile repodaki `notebooks/00_kaggle_veri_yukle.ipynb` içeriğini al (ya da hücreleri yapıştır).
2. Sağ panel: **Accelerator: None**, **Internet: On**, Secrets: `GITHUB_TOKEN` (+ `HF_TOKEN`) → *Attached*.
3. Hücreleri sırayla çalıştır. Kuru koşu şunu yazmalı: `bench 413 + train 617 = 1030`.
4. İndirme hücresi ~15–30 dk. Kesilirse aynı hücreyi tekrar çalıştır (var olanları atlar).
5. Doğrulama hücresi `SONUÇ: OK` yazmalı (1030, manifest'le eksik 0 / fazla 0).
6. *Save Version → Save & Run All (Commit)*. Bittikten sonra sürümün *Output* sekmesi → **New Dataset** → ad `disasterm3-turkiye`, **Visibility: Private**, lisans notu: CC BY-NC-SA 4.0.

## B. Doğrulama (boş notebook, tek hücre)
Yeni boş notebook → *Add Input* → kendi dataset'ini ekle; Secrets: `GITHUB_TOKEN` Attached; Accelerator None. Tek hücre:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["git","clone","-q",f"https://{tok}@github.com/MelihKotman/Deprem-VLM-RAG.git","/tmp/repo"],check=True,capture_output=True)
os.chdir("/tmp/repo")
!pip -q install -r requirements.txt
!python -m src.dm3_verify
```
Beklenen: `kullanılabilir karo ... 1030`, `SONUÇ: OK`.

## C. Sonra
`config/base.yaml → data.kaggle_dataset` alanına dataset yolunu yaz (`<kullanici>/disasterm3-turkiye`), commit/push.
