# notebooks/
Akış sırasıyla; her adım bir notebook. Çıktıları temizlenmiş halde commit edilir (büyük çıktı repoya girmez).

| Notebook | Durum |
|---|---|
| `00_kaggle_veri_yukle.ipynb` | Adım 5 (HF → Kaggle dataset; tamam) |
| `01_veri_hazirlama_disasterm3.ipynb` | hazır (Kaggle'da çalışır) |
| `02_denetim_sayfalari.ipynb` | Adım 6 (hazır; Kaggle'da çalışır) |
| `03_sizinti_ve_gecerlilik.ipynb` | Adım 7 (sızıntı kontrolü + geçerlilik maskesi; Kaggle, CPU) |
| `04_gorsel_model.ipynb` | Adım 8 kutu 3: ChangeOS pilotu (Kaggle GPU; protokol `docs/adim8_pilot_kaggle.md`) — koşuldu, pilot KAPANDI (29 Ekim) |
| `10_rag.ipynb` | Adım 10 (planlı; eski adı `06_rag`, numaralar 05–09 kullanıldığı için yeniden numaralandı) |
| `11_vlm_llm_kollar.ipynb` | Adım 11–13: kollar A, B, B+, C, D, +U (planlı) |
| `12_degerlendirme.ipynb` | Adım 14 (planlı) |

- `05_tam_egitim.ipynb` — Adım 8 tam eğitim (ön-kayıt: `docs/adim8_tam_egitim_onkayit.md`). `MODE="dry"` kuru koşu, `MODE="run"` tam eğitim (Commit modunda, iki GPU'da şeritler). Test bölmesine dokunmaz.
- `06_test_degerlendirme.ipynb` — Adım 8 test değerlendirmesi, TEK TUR (`--allow-test` yalnız burada; kapı: tüm checkpoint + tetikleyici sonucu girdi olarak ekli). Rapor: `src/test_report.py`.
- `07_katlama_olasiliklari.ipynb` — Adım 8B-1 (ön-kayıt `docs/adim8b_onkayit.md`): Osmaniye-dışarıda mevcut ağırlıkla değerlendirme + Adıyaman-dışarıda / Malatya-dışarıda yeni katlamalar; hepsinde `--dump-components` ile bina bileşeni olasılıkları (`src/dm3_probs.py`). Test bölmesine dokunmaz.
- `08_k0_geometri.ipynb` — Adım 8B / K0 geometri hızlı ölüm testi (kalıntı ofset; Kaggle CPU; ön-kayıt `docs/adim8b_k0.md`). Sonuç: kapı geçmedi (AUC 0,542 / 0,508). Etiket `adim8b-k0-v1`.
- `09_k1_golge.ipynb` — Adım 8B / K1 gölge denemesi (Kaggle CPU; ön-kayıt `docs/adim8b_k1.md`). Sonuç: kapı geçmedi (Adıyaman 0,555 / Malatya 0,606). Etiket `adim8b-k1-v1`. 8B kapanış özeti: `docs/adim8b_sonuc.md`.
