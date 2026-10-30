# Adım 8B / K0 — geometri hızlı ölüm testi (uygulama notu, 30 Ekim 2026; sonuç görülmeden yazıldı)

Ön-kayıt: `docs/adim8b_onkayit.md` §3 (K0) ve §4 (Kalıntı ofset). Bu not yalnız ön-kayıtta açık bırakılan uygulama ayrıntılarını SONUÇTAN ÖNCE sabitler.

## Sabitlenen ayrıntılar
- **Örneklem:** S1 eğitim, Adıyaman ve Malatya; her şehirde karolar `tile` sırasına dizilip `numpy.default_rng(42).permutation` ile karıştırıldı, ilk 60 alındı; hasarlı∪yıkık < 100 olsaydı permütasyonda devam edilecekti (gerekmedi: Adıyaman 268, Malatya 117). Dosya `data_manifest/geom_k0/k0_orneklem_v1.csv` (sha1 `e1a0c637ca`, 120 karo).
- **Şehir kayması:** şehrin TÜM S1-eğitim karolarının manifest `dx_post, dy_post` medyanı (tamsayıya yuvarlanmış); sonrası görüntü `read_tile(shift=…)` ile hizalanır. Birim yön u = bu vektör / norm.
- **Bina:** her sınıf maskesinin 8-bağlantılı bileşeni; alan < 40 px atılır; kontur piksellerinin ≥ %90'ı geçerlilik maskesinde değilse atılır.
- **Gradyan:** gri → Gauss(σ=1) → Sobel(3×3) büyüklüğü. Arama: (dx,dy) ∈ [−24,24]², adım 2, kontur pikselleri üzerinde gradyan toplamı (TM_CCORR); eşitlikte ilk indeks.
- **İşaret (hipotez yönü, sonuçtan önce):** şehir medyan kayması tipik ayakta binanın yaslanmasını içerir; alçalan/yıkılan bina daha az yaslanır → u yönündeki kalıntı küçük/negatif. K0 skoru = **−proj**, pozitif sınıf = hasarlı ∪ yıkık. AUC > 0,5 hipotez yönüdür. Ters yönde güçlü sapma (AUC < 0,45) H1'i desteklemez; ayrıca raporlanır, K0 kararında "geçmedi" sayılır.
- **K0 kararı (ön-kayıtla aynı):** iki şehirde de AUC(−proj) < 0,55 → H1 düşer, Kapı E koşulmaz. Aksi → Kapı E (üç şehir, tüm karolar) koşulur.
- **Ek (karar kapısı DEĞİL, yalnız betimleyici):** (a) bina alanı tertillerinde AUC; (b) oturma skoru AUC'si (doku kaybı, H1 kanıtı sayılmaz); (c) **pre_kontrol:** aynı prosedür ÖNCESİ görüntüde (etiketle hizalı olmalı) → ölçüm sağlığı; pre'de proj dağılımı ≈ 0 çevresinde dar değilse ölçüm yöntemi güvenilmez ve K0 sonucu "belirsiz" yazılır; (d) karo bootstrap %95 AA (2000).
- Sentetik selftest: `src/selftest_geom.py` (bilinen ofsetler 12/12 geri bulundu).

## Sınırlılıklar
Gradyan toplamı yakın kenarlara (komşu bina, gölge) kayabilir; yıkık binada doku bozulur; şehir kayması etiketten ölçülmüş tek vektördür; tek yapay zekâ analisti.
