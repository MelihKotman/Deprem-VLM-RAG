# Adım 8 · Kutu 3 — Gerçek veri pilotu: kod, protokol, ön-kayıt — 29 Ekim 2026

> Protokol (aşağıdaki "Ön-kayıt") hiçbir gerçek veri koşusu yapılmadan, 29 Ekim 2026'da yazıldı. Sayılar sonuçlara bakıp değiştirilmez; değişirse neden + tarih altına eklenir. Durum: kod bulutta SENTETİK karolarla doğrulandı (gerçek veri + GPU ile HENÜZ çalıştırılmadı).

## 1. Kod haritası
| Dosya | İş |
|---|---|
| `src/dm3_dataset.py` | karo okuma (öncesi/sonrası + 3 bina maskesi → tek hasar haritası 0..3), bölme CSV'si + geçerlilik, kilitler, eğitim kırpımı + artırım, tam-karo eval |
| `src/dm3_eval.py` | 3 sınıflı ortak nesne oylaması; katı / d=5 / d=15 bina bileşeni F1; piksel IoU; karo düzeyi; tetikleyici r |
| `src/train_changeos.py` | eğitim (fp16 autocast + GradScaler), `--resume`, `--max-minutes`, `--pilot`, `--eval-only`, `metrics.json`, `timing.json` |
| `src/summarize_runs.py` | koşuları tek tabloda toplar |
| `src/synth_tiles.py`, `src/selftest_eval.py`, `src/selftest_dataset.py` | sentetik karo üretici + birim testleri (gerçek veri gerekmez) |
| `notebooks/04_gorsel_model.ipynb` | Kaggle'da pilotu sırayla çalıştıran hücreler |

## 2. Teknik kararlar (kodda; itiraz yoksa geçerli)
- **Etiket:** 0 arka plan, 1 sağlam, 2 hasarlı, 3 yıkık; çakışmada ağır hasar kazanır. Eksik maske dosyası = o sınıftan bina yok. Konum = hasar > 0.
- **Geçerlilik:** `tile_validity.valid_mask` (eğitimde geçersiz piksel hasar ve konum hedefi 255 = ignore; değerlendirmede tahmin ve referans yalnız geçerli bölgede sayılır).
- **Girdi:** RGB, ImageNet ortalama/std, öncesi+sonrası 6 kanal; SAR kullanılmaz.
- **Eğitim kırpımı:** 512 px, karo başına 4 kırpım; kırpımların `dam-p` (varsayılan 0,5) oranı hasarlı/yıkık piksel çevresinden; flip + rot90; bağımsız ±%10 parlaklık/kontrast; **sonrası kırpımı ±`post-shift` px rastgele ötelenir (etiket öncesine bağlı kalır)**. Öteleme ayrı rastgele akıştan çekilir (diğer artırımları değiştirmez).
- **Kayıp:** konum BCE + Tversky(0,9/0,1); hasar CE (isteğe bağlı sınıf ağırlığı) + Dice; hepsi float32'de, ignore=255. ChangeOS'un kendi kayıp koduyla aynı terimler; tek fark ignore'un konuma da uygulanması ve CE ağırlığı. Bizim logit yolunun `ChangeOS.forward` ile aynı çıktıyı verdiği test edildi.
- **Optimizasyon:** AdamW (lr 1e-4, wd 1e-2), 1 epoch ısınma + kosinüs; tek GPU, batch 8 kırpım. Model seçimi yok: sabit epoch, son ağırlık.
- **Nesne oylaması (tüm modeller için ortak son işlem):** konum olasılığı > 0,5 → 8-komşuluk bileşenler → her bileşende sınıf 1–3 olasılık ortalaması → argmax. Ağırlık yok (xBD'ye sabit 8/38/25/11 kullanılmaz). Sınıf önsel düzeltmesi bilerek yok (seçim yapmak gerekirse ön-kayıtlı tek ayar olarak sonradan eklenir, test görülmeden).
- **Ölçütler:** `docs/adim7_olcut_tanimlari.md` + `docs/adim8_tetikleyici.md`. Uygulama notları / SAPMA: (a) toleranslı ölçütte her piksel **en yakın** referans bileşene atanır (çakışan genişletmelerde çift sayım olmasın); (b) bileşende hiç tahmin yoksa "kaçırılan" (F1'de FN), referans bölgesiyle örtüşmeyen tahmin bileşeni "yanlış alarm" (FP, sınıfı = çoğunluk sınıfı); (c) karo düzeyi: "yıkık bina var mı" F1'i + hasar yoğunluğu MAE/Spearman.
- **Kilitler (ön-kayıt koruması):** test bölmesi `--allow-test` olmadan açılmaz; Osmaniye katlaması `--trigger-fold` olmadan açılmaz; kontrol (Gaziantep) eğitimde yok.

## 3. Ön-kayıt: pilot protokolü
**Amaç:** (i) hat doğru ve hızlı mı, (ii) ölçülmüş süre/bellekle kota planı, (iii) iki serbest eğitim ayarını seçmek: sonrası öteleme aralığı ve sınıf dengesi; (iv) kodlayıcı (r50 / swint).
**Ayarlama katlaması:** S1 eğitim şehirlerinden **Malatya dışarıda** (eğitim Adıyaman + Osmaniye). Test (Antakya, e6, e7), S2 test ve **Osmaniye katlaması (tetikleyici) pilotta kullanılmaz**. Tek seed (42).
**Karar ölçütü:** birincil = Malatya'da **toleranslı (d=15) hasar-F1** (nokta tahmin); ikincil = katı hasar-F1, piksel mIoU-3. Birincil ölçüt iki seçenek arasında **0,02 mutlaktan** az farklıysa "eşit" sayılır ve **daha basit** seçenek alınır (öteleme: 0 < 16 < 32; örnekleme: `dam-p 0` < `dam-p 0,5` < `+ağırlık`; kodlayıcı: r50 < swint). Ölçütte payda 0 / tanımsız ise o koşu elenir ve nedeni yazılır.
**Aşamalar:** 0) hat testi (`--pilot`, 64 karo, 3 epoch) → süre/bellek/veri-bekleme, kayıp düşüyor mu; 1) öteleme ∈ {0, 16, 32} px, 20 epoch; 2) en iyi ötelemede `dam-p 0` / `dam-p 0,5` / `dam-p 0,5 + CE ağırlığı 1,1,3,3`; 3) süre kalırsa en iyi ayarda `--encoder swint`. Epoch sayısı: ilk yazımda 12; **Aşama 0 ölçümünden sonra 20'ye çıkarıldı (bkz. §5)**.
**Geçmeme (durma) koşulları:** kayıp NaN/sonsuz (kod durur); Aşama 0'da epoch süresi > 400 sn (kota planı yeniden); veri bekleme oranı > %40 (önbellek/`workers` düzeltmesi; eğitim kodu değişmez).
**Kota bütçesi (kaba; Aşama 0 ölçümüyle değişir):** Aşama 1–2 ≈ 5 koşu × ~25 dk ≈ 2 saat; swint +25 dk. Haftalık 30 saatin çok altında.
**Sınırlılıklar (rapora yazılacak):** Malatya katlamasında etiket kayması yüksek (medyan ≈34 px; eğitimde Adıyaman ≈45, Osmaniye ≈14) → mutlak skorlar düşük olabilir, yalnız göreli sıralama bilgisi; tek seed, kısa epoch → 0,02 eşiği altındaki farklar gürültü; seçimler Osmaniye katlamasının dolaylı olarak da eğitim şehri olmasından etkilenir (ayarlama Osmaniye etiketleriyle eğitir, tetikleyici ölçümünde Osmaniye değerlendirme tarafında kalır; tetikleyici ölçümü için model sıfırdan Adıyaman+Malatya ile eğitilir); sentetik testler modelin gerçek veride öğrendiğini kanıtlamaz.

## 4. Sentetik doğrulama sonuçları (bulut, CPU, 29 Ekim 2026; gerçek veri YOK)
- `selftest_eval`: kusursuz tahmin F1=1; küçük bina + 25 px kayma → katı 0 / d=5 0 / d=15 1 (r=0); sınıf hatası, kaçırılan, yanlış alarm, geçerlilik maskesi, oylama, tetikleyici bölgeleri geçti.
- `selftest_dataset`: bölme + kilitler, kırpım–etiket hizası, 255 ignore, önbellek eşitliği, ötelemenin yalnız sonrası görüntüyü değiştirmesi, logit yolu = `ChangeOS.forward` geçti.
- 14 epoch'luk mini koşu (6 sentetik karo): kayıp düştü (c_ce 1,18→0,10; konum IoU 0,77 hold-out karoda). Hız/bellek ölçümü anlamsız (CPU).

## 5. Aşama 0 sonucu ve ön-kayıt güncellemesi (29 Ekim 2026; Aşama 1 sonuçlarına bakılmadan yapıldı)
**Ölçüm (Kaggle T4, fp16, 64 karo, crop 512, batch 8 kırpım, 4 çalışan, önbellekli):** sabit-durum epoch (2.–3.) 21,9–22,5 sn ≈ **11,5 kırpım/sn** (sentetik testteki 11,9 ile uyumlu → GPU-bağlı); veri bekleme %2 (1. epoch %5,5: önbellek dolumu); tepe bellek **7,35 GB** (yalnız 1. epoch'ta 15,06 GB: cudnn.benchmark çalışma alanı, T4 sınırına yakın; çökmedi); kayıp düştü, NaN yok (3 epoch: c_ce 1,07→0,40; t1_bce 0,54→0,29); değerlendirme 0,62 sn/karo. Betik çıktısındaki "150 sn/epoch, 2,09 saat" tahmini 1. epoch'u ortalamaya kattığı için fazlaydı; kod düzeltildi (sabit-durum ortalaması). Düzeltilmiş tahmin: Malatya katlaması eğitimi (320 karo) ≈ **111 sn/epoch**.
**Durma koşulları:** epoch ≤ 400 sn → SAĞLANDI (111); veri bekleme ≤ %40 → SAĞLANDI (%2); kayıp sonlu → SAĞLANDI. Aşama 0 GEÇTİ (hat çalışıyor). Not: Aşama 0 metrikleri (sanity/Malatya, 3 epoch) karar vermez; kullanılmadı.
**Ön-kayıt değişikliği (neden: ölçülen hız kotaya bol yer bırakıyor; 12 epoch ≈ 3,8 bin adım nadir hasar sınıflarını öğrenmek için kısa kalabilir; Aşama 1 sonucu görülmeden):**
- Aşama 1–3 epoch sayısı **12 → 20** (koşu başı ≈ 37 dk + ~2 dk değerlendirme; 1. koşuda +önbellek dolumu). Bundan sonra değişmez.
- Karar ölçütü, 0,02 eşiği, basitlik sırası, Malatya katlaması DEĞİŞMEDİ.
- Aşama 2'de `dam-p 0,5` koşusu ayrıca yapılmaz: Aşama 1'in kazanan ötelemesi `dam-p 0,5` ile koşulmuştur, o koşu referanstır (Aşama 2: `dam-p 0` ve `0,5 + ağırlık`).
**Kota planı (ölçüme dayalı; tam boyutta ölçülmedi, doğrusal ölçekleme):** pilot 5 koşu ≈ 3,2 saat + swint (≈1,4× yavaş varsayımı, ÖLÇÜLMEDİ) ≈ 0,9 saat. Tam eğitim 50 epoch: S1 (474 karo) ≈ 164 sn/epoch ≈ **2,3 saat/seed** → 3 seed ≈ 6,8 saat; S2 (618 karo) ≈ **3,0 saat** (tek seed); tetikleyici katlaması (Adıyaman+Malatya, 306 karo) ≈ 1,5 saat. Toplam ≈ 11,3 saat + pilot ≈ 4,1 saat ≈ **15,4 saat < 30 saat/hafta**. İki GPU'yu paralel kullanmak duvar saatini yarılar; kota sayacının GPU başına mı oturum başına mı işlediği doğrulanmadı.
**Risk:** 1. epoch 15,06 GB tepe → tam boyutta farklı bir ayırma düzeni OOM verirse `--no-cudnn-benchmark` ile yeniden koş (hız biraz düşebilir).

## 6. Aşama 1 sonucu, tanı ve ön-kayıt güncellemesi (29 Ekim 2026; tanı çıktıları görülmeden yazıldı)
**Aşama 1 sonucu (Malatya katlaması, r50, seed 42, `dam-p 0,5`, YALNIZ 12 epoch):** (SAPMA: koşular notebook'un eski sürümüyle 12 epoch koşuldu; §5'te 20'ye çıkarılmıştı. Üç koşu kendi aralarında karşılaştırılabilir; karar bunun üzerinden verilmedi.)
| öteleme | katı = tol5 = tol15 hasar-F1 | katı macro-F1 | piksel mIoU | konum IoU |
|---|---|---|---|---|
| 0 | 0,082 | 0,339 | 0,261 | 0,776 |
| 16 | 0,038 | 0,307 | 0,251 | 0,778 |
| 32 | 0,060 | 0,324 | 0,255 | 0,782 |
Kuralın mekanik sonucu: öteleme 0 (0,082 − 0,060 = 0,022 > 0,02; ayrıca en basit). **Bu sonuç KESİN DEĞİL:** (i) hasar-F1 ≤ 0,08, gürültü tabanında; (ii) katı = tol5 = tol15 (3 hanede aynı) → ölçüt ayırt edici değil; konum IoU'su 0,78 iken hasar sınıfları neredeyse hiç tahmin edilmiyor. Olası nedenler (ayrıştırılacak): (1) oylamada sınıf önseli yok (sağlam ≈ %97 → ortalama olasılıkta sağlam hep kazanır; ChangeOS'un kendi oylaması bu yüzden sabit ağırlıklıdır); (2) eğitimde hasar öğrenilmedi (12 epoch, nadir sınıf); (3) Malatya etiketi/sonrası görüntü kayması (medyan ≈34 px: konum başı öncesinden, hasar başı sonrasından besleniyor → hasar kanıtı yanlış piksele düşer; d = 15 tahmin kaymasını düzeltir, girdi kaymasını düzeltmez).
**Tanı (Aşama 1b, yeniden eğitim yok):** (a) karışım matrisi + kademe kırılımı; (b) eğitim-içi değerlendirme (`--eval-in-sample`); (c) oylama varyantları `none` / `sqrt_inv` / `inv` (ağırlık = eğitim bina sayılarından N_sağlam/N_c, `inv` ya da kökü; Malatya sayıları kullanılmaz; test görülmez).
**Karar kuralları (ön-kayıt, tanı sonucu görülmeden):**
1. Oylama: `p1_shift0` ağırlıklarıyla üç varyant karşılaştırılır; birincil ölçüt tol15 hasar-F1, 0,02 eşiği, basitlik sırası `none` < `sqrt_inv` < `inv`. Seçilen oylama TÜM sonraki değerlendirmelerin ortak son işlemi olur (ve tüm modellere uygulanır).
2. Seçilen oylamayla `p1_shift16` ve `p1_shift32` yeniden DEĞERLENDİRİLİR (eval-only) ve §3'teki öteleme kuralı uygulanır. Aşama 1 yeniden eğitilmez.
3. Eğitim-içi hasar-F1 (katı) < 0,30 ise model eğitimde de hasarı öğrenemiyor → Aşama 2 (`dam-p 0` / ağırlık) beklemeden yürür, epoch 20'ye çıkar; ≥ 0,30 ama Malatya düşükse sorun genelleme/kayma → Aşama 2 durur, girdi kayma düzeltmesi (manifest `dx_post/dy_post` ile ön hizalama; seçenek (c)) Melih'le tartışılır — bu, 28 Ekim'de seçilen (b)'yi değiştirir, onay gerekir. (0,30 eşiği benim önerim, onay bekliyor.)
4. Tanıdan sonra bile katı hasar-F1 < 0,15 kalırsa pilot "ölçüt ayırt edici değil" diye kapatılır, ayar seçimi yapılmaz; varsayılanlarla (öteleme 0, `dam-p 0,5`, r50) tam eğitime geçilip sınırlılık yazılır.

## 7. Aşama 1b sonuçları, kuralların uygulanması ve Aşama 2 ön-kaydı (29 Ekim 2026; Aşama 2 sonuçları görülmeden)
**Tanı çıktıları (Malatya katlaması, `p1_shift0` YENİDEN eğitildi: 12 epoch, aynı ayarlar; eski koşunun dosyaları oturum kapanınca kayboldu).**
| oylama | tol15 hasar-F1 | katı macro-F1 | yıkık F1 | hasarlı F1 |
|---|---|---|---|---|
| none | 0,013 | 0,290 | 0,026 | 0,000 |
| sqrt_inv | 0,094 | 0,344 | 0,189 | 0,000 |
| inv | 0,084 | 0,331 | 0,127 | 0,041 |
Eğitim-içi (40 eğitim karosu, oylama none): katı hasar-F1 0,280 (hasarlı 0,194, yıkık 0,367). Malatya referansı: 8.376 sağlam, **105 hasarlı, 76 yıkık** bina bileşeni → hasar-F1 ~180 bina üzerinden. Katı = tol5 = tol15 hasar-F1 (tahmin edilen hasar neredeyse yok, tolerans bir şey değiştirmiyor); T1 3 karo, T2 66, T3 85 (kademe kırılımı bilgi).
**Gürültü bulgusu (önemli):** aynı ayar (öteleme 0, 12 epoch, seed 42, oylama none) iki koşuda katı hasar-F1 **0,082 ve 0,013** verdi. Tek seed + ~180 hasarlı bina → koşular arası fark ≈ 0,07. Aşama 1'in (öteleme 0/16/32: 0,082/0,038/0,060) sıralaması bu yüzden **bilgi taşımıyor**; 0,02 eşiği gürültünün çok altında. Eski checkpoint'ler yok → shift16/32 yeniden değerlendirilemez (kural 2 uygulanamadı).
**Kuralların uygulanması (§6):**
1. Oylama: sqrt_inv ile none farkı 0,081 > 0,02 (aynı checkpoint, eşleşik karşılaştırma → gürültüden bağımsız); sqrt_inv ile inv farkı 0,010 < 0,02 → eşit → basit olan **sqrt_inv SEÇİLDİ**, `--vote` varsayılanı yapıldı; tüm modellere ortak son işlem. (Tek checkpoint, ~180 hasarlı bina; sınırlılık.)
2. Uygulanamadı (checkpoint yok) ve gürültü nedeniyle anlamsız → öteleme seçimi Aşama 1'den ÇIKARILDI.
3. Eğitim-içi katı hasar-F1 0,280 < 0,30 → "eğitimde öğrenememe" dalı: Aşama 2 yürür, epoch 20. (0,30 eşiğine çok yakın ve eğitim-içi örnek ~46 hasarlı bina → kesin değil; kayma/girdi hizalaması hâlâ aday: kural 3'ün ikinci dalı Aşama 2 sonrası yeniden bakılır.)
4. Kural 4'ün (katı hasar-F1 < 0,15 → pilot kapanır) kural 3 ile çakıştığı görüldü (yazım belirsizliği benim): **yorum: kural 4 Aşama 2 SONUNDA uygulanır** (20 epoch, sqrt_inv ile en iyi koşu < 0,15 ise pilot "ayırt edici değil" diye kapanır ve varsayılanlarla tam eğitime geçilir).
**Aşama 2 (güncel):** 20 epoch, oylama sqrt_inv, Malatya katlaması, seed 42; 4 koşu: `p2_ref` (öteleme 0, `dam-p 0,5`), `p2_dam0`, `p2_w1133` (CE ağırlığı 1,1,3,3), `p2_shift16`; her koşu için eğitim-içi kontrol. **Karar eşiği 0,02 → 0,07** (gözlenen koşu-arası gürültüden; Aşama 2 sonuçları görülmeden): birincil ölçüt tol15 hasar-F1; bir seçenek `p2_ref`ten 0,07'den fazla iyiyse seçilir, değilse varsayılan (`dam-p 0,5`, ağırlıksız CE, öteleme 0, r50) kalır. Kodlayıcı (swint) karşılaştırması bu gürültüyle anlamsız → pilottan çıkarıldı, r50 sabit. Tam eğitimde gürültü için seed ≥3 zaten planlı.

## 8. Aşama 2 sonucu, pilotun kapanışı ve Aşama 3 (hizalama tanısı) ön-kaydı (29 Ekim 2026; Aşama 3 sonuçları görülmeden)
**Aşama 2 (Malatya, 20 epoch, oylama sqrt_inv, seed 42), katı / tol15 hasar-F1:** `p2_ref` 0,000 / 0,000; `p2_dam0` 0,012 / 0,012; `p2_w1133` 0,044 / 0,034; `p2_shift16` 0,025 / 0,025. Eğitim-içi (40 karo) katı: ref 0,225; dam0 0,191; w1133 0,304; shift16 0,302 (n ≈ 46 hasarlı bina, gürültülü).
**Kuralların uygulanması:** hiçbir seçenek `p2_ref`ten 0,07'den fazla iyi değil (en iyi fark +0,034) → varsayılanlar kalır: öteleme 0, `dam-p 0,5`, ağırlıksız CE, r50, oylama sqrt_inv. Kural 4: en iyi koşu tol15 hasar-F1 0,034 < 0,15 → **ayar seçimi için pilot kapandı ("ölçüt ayırt edici değil": Malatya'da ~180 hasarlı bina, koşu arası gürültü ≈ 0,07, aynı ayarla 12 epoch'ta 0,094, 20 epoch'ta 0,000).** Epoch 12 → 20 Malatya'yı iyileştirmedi, eğitim-içi de iyileşmedi (0,280 → 0,225, ikisi de gürültü içinde) → sorun salt "az eğitim" değil.
**Neden Aşama 3:** hasar, eğitim karolarında bile 0,19–0,30; tam eğitim (≈ 11 saat) bu hâliyle büyük olasılıkla düşük hasar skoru verir. En olası tek-neden adayı girdi hizası: etiketler öncesine hizalı, sonrası görüntü Adıyaman'da ≈45, Malatya'da ≈34, Osmaniye'de ≈14 px kaymış (manifest) → hasar başı sonrasından, yanlış piksel komşuluğundan besleniyor. Bu, 28 Ekim'deki "eğitimde öteleme artırımı (b)" kararının varsaydığından farklı: (b) kaymayı büyütüyor, düzeltmiyor.
**Aşama 3 (tek deney, etki büyüklüğü gürültünün çok üstünde beklenir):**
- (a) ücretsiz: `p2_ref`i TÜM 320 eğitim karosunda şehir (Adıyaman / Osmaniye) ve kayma kademesine göre değerlendir. Beklenti (hizalama hipotezi): Osmaniye ve T1 karolarında hasar-F1 Adıyaman / T3'ten belirgin yüksek.
- (b) `p3_oracle`: aynı eğitim (20 epoch, öteleme 0, `dam-p 0,5`, sqrt_inv) ama sonrası görüntü manifest `dx_post/dy_post` ile hizalanmış (eğitim ve Malatya değerlendirmesinde). **Oracle:** kayma ETİKET kenarından ölçülmüş (`notebooks/01`), gerçek (etiketsiz) bir yöntem değildir; yalnız "kayma nedeni mi" sorusunu yanıtlar. Hizalama sözleşmesi sentetik testte doğrulandı (öncesi–sonrası fark 0,386 → 0,000).
**Karar kuralı (ön-kayıt):** `p3_oracle` Malatya tol15 hasar-F1'i ≥ 0,15 (kural 4 eşiği) **ve** `p2_ref`ten ≥ 0,07 fazlaysa → kayma ana nedendir; tam eğitim BEKLER, etiketsiz görüntü-tabanlı hizalama (öncesi–sonrası kayıt) tasarımı Melih'le konuşulur (28 Ekim'deki (b) kararının değişmesi onay ister; test etiketinden ölçülen kayma kullanılmaz). Aksi halde → kayma ana neden değildir; varsayılanlarla tam eğitime geçilir ve düşük hasar-F1 ile tetikleyici (Osmaniye katlaması) sonucu raporlanır; "önce bina düzeyi, olmazsa karo düzeyi" kuralı (Revizyon 1) zaten bunu öngörüyor.
