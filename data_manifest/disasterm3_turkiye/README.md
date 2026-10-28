# disasterm3_turkiye

DisasterM3 (DOI 10.48550/arXiv.2505.21089; HF `Kingdrone-Junjue/DisasterM3`, CC BY-NC-SA 4.0) Türkiye alt kümesi. **Görüntüler bu bilgisayarda değil** (≈5 GB, bulut oturumunda işlendi); burada yalnız küçük tablolar var.

- `manifest.csv`: karo, event, bölge (BRIGHT eşlemesiyle), bölme (train/bench), sınıf pikselleri/bina sayıları, öncesi/sonrası/SAR var mı, `status` (ok = öncesi + sonrası + maske var), karo başına oturma skorları (`s_pre`, `s_post`) ve sonrası kayması (`dx_post`, `dy_post`, `kayma_post_px`), eşleşen BRIGHT karosu (`bright_tile`).
- `bolge_ozet.csv`: bölge düzeyinde özet.

Bulgular (27 Ekim 2026): kullanılabilir 1.030 karo (eğitim 617, benchmark 413). Event 0=Gaziantep, 1=Adıyaman, 3=Malatya, 4=Antakya/Defne, 5=Osmaniye: bina etiketleri BRIGHT ile piksel piksel aynı (752 karo eşleşti) ve sonrası görüntünün kayması bizim BRIGHT+Maxar ölçümümüzle aynı. Event 6 (197 karo, yalnız benchmark) ve 7 (84 karo) BRIGHT'la eşleşmedi, kırsal ağırlıklı, SAR yok, kayma düşük.
