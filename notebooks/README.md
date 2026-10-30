# notebooks/
Akış sırasıyla; her adım bir notebook. Çıktıları temizlenmiş halde commit edilir (büyük çıktı repoya girmez).

| Notebook | Durum |
|---|---|
| `01_veri_hazirlama_disasterm3.ipynb` | hazır (Kaggle'da çalışır; Adım 5'te dataset yoluna bağlanacak) |
| `02_denetim_sayfalari.ipynb` | Adım 6 (hazır; Kaggle'da çalışır) |
| `03_sizinti_ve_gecerlilik.ipynb` | Adım 7 (sızıntı kontrolü + geçerlilik maskesi; Kaggle, CPU) |
| `04_gorsel_model.ipynb` | Adım 8 kutu 3: ChangeOS pilotu (Kaggle GPU; protokol `docs/adim8_pilot_kaggle.md`) — kod sentetik veriyle test edildi, gerçek veriyle henüz koşulmadı |
| `05_vlm_llm.ipynb` | Adım 11–12 |
| `06_rag.ipynb` | Adım 10 |
| `07_degerlendirme.ipynb` | Adım 14 |

- `05_tam_egitim.ipynb` — Adım 8 tam eğitim (ön-kayıt: `docs/adim8_tam_egitim_onkayit.md`). `MODE="dry"` kuru koşu, `MODE="run"` tam eğitim (Commit modunda, iki GPU'da şeritler). Test bölmesine dokunmaz.
- `06_test_degerlendirme.ipynb` — Adım 8 test değerlendirmesi, TEK TUR (`--allow-test` yalnız burada; kapı: tüm checkpoint + tetikleyici sonucu girdi olarak ekli). Rapor: `src/test_report.py`.
