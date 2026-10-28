# yedek_bright_maxar_turkiye

Makale 1 veri seti: BRIGHT (Chen ve ark., ESSD 2025, DOI 10.5194/essd-17-6217-2025; veri DOI 10.5281/zenodo.14619797) Türkiye 2023 deprem hasar etiketleri + Maxar Open Data olay-sonrası optik görüntü (CC BY-NC 4.0).

- 1.114 karo, 1024x1024 (BRIGHT ızgarası, EPSG:32637). `pre/` öncesi optik, `post/` Maxar sonrası optik, `label/` etiket (0 arka plan, 1 sağlam, 2 hasarlı, 3 yıkık).
- `manifest.csv`: karo, şehir, bölme, sınırlar, sonrası tarih/off-nadir/GSD, kaynak COG, sonrası ve öncesi görüntüde boş (siyah) piksel oranı (`blank_frac`, `pre_blank_frac`), sınıf pikselleri.
- Bölme (şehir düzeyi): dev = Adıyaman, Malatya, Kahramanmaraş, İslahiye (624 karo; eğitim + şehir-bırakmalı doğrulama); test = Antakya/Defne, Osmaniye, Gaziantep (490 karo; yalnızca son değerlendirme).
- Yeniden üretmek için: `Kod/M1/01_veri_hazirlama_bright_maxar_yedek.ipynb` (yeniden başlatılabilir; eski betik kaldırıldı).

## Hizalama (kayma)
Etiketler öncesi (nadire yakın) geometriye göre çizildiği için sonrası görüntüde konturlar çatılardan kayık görünebilir (en çok Adıyaman/Malatya ≈44°). Daha düşük açılı Maxar sahnesi yok (STAC'te denetlendi). xBD ve BRIGHT'taki gibi **referans öncesi + etiket, taşınan şey sonrası görüntü**:
- `label/` ve `pre/` değişmedi. `post/` ham sonrası görüntü.
- `post_hizali/`: sonrası görüntü, şehir başına ölçülen kaymayla (`hizalama.csv`) etiketin/öncesinin geometrisine taşınmış; kaydırmayla açılan kenar şeridi siyah (0), kayıpta/metrikte dışlanmalı (`gecerli_maske`, notebook 5c). `overlay_hizali/`: aynı etiket konturu hizalı sonrası görüntü üzerinde (görsel kontrol).
- Sonrası görüntünün taşındığı miktar (piksel, `-(dx, dy)`; hizalama.csv'de dx, dy): Adıyaman (44, 32) ≈17 m; Malatya (-32, 6) ≈10 m; Gaziantep (-13, -21) ≈9 m; İslahiye (2, -15) ≈5 m; Osmaniye (-11, -9) ≈4 m; Kahramanmaraş (2, 10) ≈3 m; Antakya uygulanmadı (≈0, tutarsız).
- Yöntem: etiket konturlarının sonrası görüntü gradyanına oturması (FFT çapraz korelasyonu), şehir başına ≤100 karonun ortancası, yarı karoda kestirip diğerinde sınama. Tek öteleme bina yüksekliğine bağlı kalan hatayı gidermez; değerlendirmede bina düzeyi yerine tampon/karo düzeyi ölçüt kullanın.

## Bilinen sorunlar
- Öncesi görüntüler kaynakta (BRIGHT, nodata=0) kısmen boş: 215 karoda %5'ten, 87 karoda %50'den fazla siyah. `pre_blank_frac` ile ölçülür; karolar silinmedi.
- Sonrası görüntü ≈2× yukarı örneklenmiş (Maxar GSD ≈0,6-0,9 m, karo pikseli ≈0,3-0,35 m).
- Sonrası sahnelerin bakış açısı şehre göre değişiyor (off-nadir: Adıyaman/Malatya ≈44°, Antakya/Gaziantep ≈34°, Kahramanmaraş/Osmaniye ≈29–31°, İslahiye ≈6°). Yüksek açıda bina yatması etiketleri kaydırır; bina düzeyi kıyas toleranslı olmalı.
- Şehir bölmesi aynı zamanda bakış açısı bölmesi: genelleme farkı iki etkiyi karıştırır.
- Tek olay (6 Şubat 2023): sonuçlar "aynı olayın görülmemiş şehirleri" için geçerlidir.
- Lisans: Maxar CC BY-NC 4.0 (ticari olmayan). BRIGHT etiketleri görüntü lisansını izler; yayın öncesi kontrol edin.

## Etiketleri görüntüleme
`label/*.png` ham sınıf indeksidir (0-3, tek kanal); görüntü izleyicilerde simsiyah görünür, bu normaldir. Eğitimde ve değerlendirmede **yalnızca bunu** kullanın. Gözle bakmak için: `label_renkli/` (0 siyah, 1 yeşil sağlam, 2 sarı hasarlı, 3 kırmızı yıkık) ve `overlay/` (etiket + sonrası görüntü bindirmesi). İkisi de notebook tarafından üretilir.
