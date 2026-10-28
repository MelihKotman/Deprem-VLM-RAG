# RAGEarthQuake/ → deprem-vlm-rag/ taşıma listesi (28 Ekim 2026)

Eski klasör (`RAGEarthQuake/`) silinmedi/değiştirilmedi; aşağıdakiler **kopyalandı**.

| Eski yer | Yeni yer | Not |
|---|---|---|
| `Kod/M1/01_veri_hazirlama_disasterm3.ipynb` | `notebooks/` | Ana hat |
| `Veri/disasterm3_turkiye/{manifest.csv,bolge_ozet.csv,README.md}` | `data_manifest/disasterm3_turkiye/` | Küçük tablolar |
| `Kod/M1/01_veri_hazirlama_bright_maxar_yedek.ipynb` | `archive/bright_maxar_yedek/` | Çıktıları temizlendi (4,9 MB → küçük). Proje-kökü bulucu eski düzeni (`Kod/` + `Veri/`) bekler; yeniden çalıştırılacaksa `RAG_PROJE` ayarlanmalı |
| `Veri/yedek_bright_maxar_turkiye/{manifest.csv,hizalama.csv,README.md}` | `archive/bright_maxar_yedek/` | Görüntüler (≈1,9 GB) taşınmadı |
| `Kod/M1/README.md` (akış tablosu) | `README.md` Klasörler/Durum bölümü | Notebook 02–05 adım gelince eklenecek |

## Repo dışında kalanlar (bilerek)
- `Makaleler/` (276 MB PDF): telif; makale yazımında ayrı klasörden kullanılır. `Makaleler/README.md` künye/DOI eşlemesi orada.
- `Notlar/` (makale notları, sentez, kararlar): Projects'teki `notlar/` ile birlikte yürür; repoya kopyalanmadı (çift kopya sapar).
- `Veri/yedek_bright_maxar_turkiye/` görüntü klasörleri (pre/post/label/overlay…): yerelde kalır.
