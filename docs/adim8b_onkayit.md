# Adım 8B ön-kaydı — yükseklik-kaybı geometrisi ve kalibre olasılık + karo bağlamı (30 Ekim 2026)

Durum: YAZILDI, onaylandı (Melih, 30 Ekim: "kapılar koyarız, sen nasıl düşünüyorsan öyle olsun"). Yazıldığı anda hiçbir 8B sonucu görülmedi. Görülmüş olan: S1/S2 test v1 sonuçları (toplam tablolar) ve hoca görüşmesi; yani fikirler test sonrası doğdu → tüm 8B sonuçları "test sonrası tasarlanan, keşfedici (post hoc)" olarak etiketlenir. Kaynak notlar: `notlar/adim8-hoca-karari-egim-ikili-karo-bina.md`, `notlar/adim8-test-sonucu.md`. Plan: yol haritası Adım 8B.

## 1. Hipotezler
- **H1 (geometri):** hizalamadan sonra, sonrası görüntüde bina konturunun şehir kaymasına göre kalıntı ofseti yükseklik kaybını taşır; sağlam ile hasarlı∪yıkık arasında ayrım sağlar.
- **H2 (kalibrasyon + bağlam):** kalibre bina olasılığı F1-optimum eşikle okunduğunda ve karo/komşu bağlamı eklendiğinde ikili hasar F1'i yükselir. Kazanç iki kaynağa ayrılır: eşik ve karo bilgisi.

## 2. Veri kullanım kuralları
- Tasarım, eşik, kalibrasyon ve ikinci aşama YALNIZ eğitim şehirlerinde (Adıyaman, Malatya, Osmaniye), şehir-bırakmalı. S1 test event'leri (Antakya, e6, e7) donma öncesi açılmaz. S2 kullanılmaz.
- Gaziantep (hasarsız kontrol) yalnız yanlış-alarm raporu için, en sonda.
- Etiketler silinmez; ikili görünüm: sağlam = 0, hasarlı ∪ yıkık = 1. Bölmeler `split_S1_v1` donuk.

## 3. Sıra ve kapılar (en ucuz kapı önce; her biri ölüm anahtarı)
**K0 — hızlı ölüm testi (geometri; model eğitimi yok).** Adıyaman ve Malatya'dan (44°) her birinden 60 rastgele karo (seed 42; bu karolarda her şehirde en az 100 hasarlı∪yıkık bina). Kalıntı ofset (§4) ile sağlam vs hasarlı∪yıkık AUC. İki şehirde de AUC < 0,55 ise H1 düşer, tam Kapı E koşulmaz.
**Kapı E — geometri ölçülebilirliği (üç eğitim şehri, tüm karolar).** Her şehirde AUC ≥ 0,65 VE bina alanı tertillerinin en az ikisinde aynı yönde AUC > 0,55. Pozitif sayısı < 100 olan şehir tek başına geçiremez/düşüremez (yön olarak raporlanır); karar üç şehrin havuzlanmış sonucu + diğer iki şehrin yönü ile. Geçmezse H1 reddedilir, "ölçülebilir değil" negatif bulgu yazılır.
**8B-1 — tanı (yeni mimari YOK).** Şehir-bırakmalı katlamalar: Osmaniye-dışarıda (VAR; tetikleyici checkpoint), Adıyaman-dışarıda ve Malatya-dışarıda (YENİ; ChangeOS, ön-kayıtlı varsayılanlar, seed 42, 50 epoch). Yalnız dışarıdaki şehir olasılıkları geçerli (eğitim-içi iyimser). Çıktı: bina bileşeni başına sınıf olasılıkları + karo skorları, olasılık histogramı, güvenilirlik diyagramı ve ECE.
**Kapı T — kararsız bant.** Havuzlanmış dışarıdaki şehir olasılıklarında ikili hasar olasılığı p ∈ [0,3; 0,7] bandına düşen gerçek hasarlı∪yıkık bina payı ≥ %20. Geçmezse karo-bağlam kısmı (3b) düşer; yalnız kalibre eşik (ii) kalır.
**Kapı F — karo bağlamı katkısı.** (iv) veya (v) kolu, (ii)'ye göre: eşleştirilmiş mutlak ΔF1 ≥ 0,03 VE karo-bootstrap %95 AA 0'ı dışlıyor, en az 2 şehirde. (0,03 ≈ eğitim seed gürültüsünün altındaki en küçük anlamlı etki için seçildi; S1 test seed std ≈ 0,04–0,07.) Geçmezse "karo bilgisi katkısız, kazanç eşikten" yazılır.

## 4. Yöntem ayrıntıları (sonuçlardan önce sabit)
**Kalıntı ofset (K0, Kapı E).** Kontur: etiket bina poligonu (öncesi, zemin referansı). Görüntü: `post_hizali`. Bina başına, kontur piksellerinin sonrası görüntü gradyan büyüklüğü toplamını maksimize eden (dx, dy) ∈ [−24, 24]² px (adım 2 px), şehir medyan kaymasından sonra. Kalıntı vektör, şehir medyan kayma vektörünün birim yönüne izdüşürülür (görüş azimutu metadatası yoksa). Sınıf etiketi ofset hesabında kullanılmaz; AUC en sonda hesaplanır. Ayrı sütunlar: izdüşümlü ofset, oturma skoru (gradyan toplamı).
**Dürüstlük uyarısı:** yıkık binada kenar dokusu bozulduğundan oturma skoru kendiliğinden düşer; skor farkı "doku kaybı"dır, geometri değil. H1 yalnız izdüşümlü ofsetle sınanır; oturma skoru ayrı raporlanır ve H1 kanıtı sayılmaz.
**Parametrik model.** Ayakta binada beklenen kayma ≈ h·tan(θ); kalıntı r = (h_ref − h)·tan(θ)·u. h bilinmediğinden bina alanı tertilleri içi z-skor.
**Kalibrasyon ve eşik (iç içe, sızıntısız).** Üçüncü şehrin olasılıkları, diğer iki şehrin dışarıdaki-şehir olasılıklarında öğrenilen kalibrasyonla (sıcaklık/isotonik) düzeltilir; eşik, aynı iki şehirde F1'i maksimize eden değerdir. Kontrol: kuramsal F1-optimum eşik ≈ F*/2 (Lipton ve ark. 2014, arXiv:1402.1892, DOI 10.48550/arXiv.1402.1892).
**İkinci aşama (yalnız ilgili kapı geçerse).** Lojistik regresyon (gerekirse küçük GBM). 3a girdi: kalibre p_bina, kalıntı ofset (alan-tertil z-skor), bina alanı. 3b: + p_karo, komşu bina ortalama p (yarıçap r px, 8B-3 başında sabitlenir), karo hasar yoğunluğu. Aynı şehir-bırakmalı iç içe düzen.
**Ablasyon kolları:** (i) çıplak argmax (mevcut) · (ii) yalnız bina, kalibre + F1-optimum eşik · (iii) +geometri · (iv) +karo · (v) +geometri+karo · (vi) oracle karo (gerçek karo etiketi; üst sınır, yöntem değil).

## 5. Ölçütler ve istatistik
- Birincil: bina düzeyi ikili hasar var/yok F1 (katı); ikincil: tol15, kesinlik, duyarlılık, hasar yoğunluğu Spearman, AUC-PR.
- Havuzlanmış dışarıdaki şehir sonuçları + şehir bazlı tablo. Eşleştirilmiş fark, karo bazlı bootstrap (2.000 yeniden örnekleme) %95 AA. Üç şehir → küme bootstrap yok, şehir tablosu + yön tutarlılığı.
- Gaziantep kontrol: yanlış alarm (sağlam bina hasar işaretli oranı, karo sayısı); eşik düşünce artması beklenir, mutlaka raporlanır.
- Fold eğitimi tek seed (42): eğitim varyansı ölçülmez, sınırlılık.

## 6. Test kararı (donma sonrası)
Yöntem donar (git etiketi `adim8b-v1`, yapılandırma sha). Sonra BİR kez, ÖNCEDEN şöyle: S1 seed 42/43/44 checkpoint'leri 396 test karosunda çıkarım yapar (bu ikinci test okumasıdır; yeniden eğitim yok); donmuş kalibrasyon/eşik/ikinci aşama uygulanır; aynı karolarda temel çizgiye (i) eşleşik karşılaştırma; "post hoc" etiketi. Kalibrasyon aktarımı varsayımı: iki şehirle eğitilmiş katlama modellerinde öğrenilen kalibrasyon/eşik/ikinci aşama, üç şehirle eğitilmiş S1 modellerine uygulanır → dağılım uyuşmazlığı riski, sınırlılık olarak yazılır. Alternatif (test hiç açılmaz) yalnız Melih önceden isterse.

## 7. Durdurma ve zaman kutusu
- K0/Kapı E başarısız VE Kapı T başarısız → 8B kapanır; yalnız (ii) kalibre eşik B kolunda kalır (kısa negatif bulgu notu).
- Biri geçerse yalnız o kol sürer.
- Zaman kutusu: 7 gün (6 Kasım 2026'ya kadar); aşılırsa o ana kadarki sonuç dondurulur ve olduğu gibi raporlanır.
- ALES (29 Kasım), EVREN Kaşifi başvurusu (30 Kasım), EVREN ücretsiz pencere 1 Aralık: 8B bu takvimi bloklamaz; Adım 9–12 paralel yürür.

## 8. Başarısızlık da raporlanır
Her kapının sonucu geçsin ya da geçmesin tek tabloda yazılır. Sonuç beklenene uymazsa kapı veya eşik değiştirilmez; sapma tarihle belgelenir.

## 9. Önerilen sohbet sırası
1. **8B-1 (kod + Kaggle):** olasılık dökümü, iki yeni katlamayı Commit modunda başlat (arka planda ≈3–4 sa GPU; kota Cmt 31 Ekim 03:00'te sıfırlanır).
2. **K0 / Kapı E (katlamalar koşarken):** CPU/hafif GPU; ofset ölçümü.
3. **8B-3:** kapılara göre ikinci aşama + ablasyon.
4. **8B-4:** donma, test kararı, kapanış.
Açılış cümlesi: "yol-haritasi-checklist.md, Adım 8B-1'e başlıyoruz."

## 10. Bilinen sınırlılıklar (şimdiden)
Test sonrası tasarım; Osmaniye'de az hasarlı bina (kalibrasyon yön gösterir); tek seed katlamaları; geometri gürültüsü (hizalama saçılımı 15–19 px, tek/çok katlı yükseklik çeşitliliği); kalibrasyon aktarımı varsayımı; tek yapay zekâ sayıcılı etiket denetimi.
