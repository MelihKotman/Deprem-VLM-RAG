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
**Aşamalar:** 0) hat testi (`--pilot`, 64 karo, 3 epoch) → süre/bellek/veri-bekleme, kayıp düşüyor mu; 1) öteleme ∈ {0, 16, 32} px, 12 epoch; 2) en iyi ötelemede `dam-p 0` / `dam-p 0,5` / `dam-p 0,5 + CE ağırlığı 1,1,3,3`; 3) süre kalırsa en iyi ayarda `--encoder swint`. Epoch sayısı Aşama 0'daki ölçüme göre kotaya sığacak biçimde (12'den düşürülmeden) ayarlanır; değişirse buraya yazılır.
**Geçmeme (durma) koşulları:** kayıp NaN/sonsuz (kod durur); Aşama 0'da epoch süresi > 400 sn (kota planı yeniden); veri bekleme oranı > %40 (önbellek/`workers` düzeltmesi; eğitim kodu değişmez).
**Kota bütçesi (kaba; Aşama 0 ölçümüyle değişir):** Aşama 1–2 ≈ 5 koşu × ~25 dk ≈ 2 saat; swint +25 dk. Haftalık 30 saatin çok altında.
**Sınırlılıklar (rapora yazılacak):** Malatya katlamasında etiket kayması yüksek (medyan ≈34 px; eğitimde Adıyaman ≈45, Osmaniye ≈14) → mutlak skorlar düşük olabilir, yalnız göreli sıralama bilgisi; tek seed, kısa epoch → 0,02 eşiği altındaki farklar gürültü; seçimler Osmaniye katlamasının dolaylı olarak da eğitim şehri olmasından etkilenir (ayarlama Osmaniye etiketleriyle eğitir, tetikleyici ölçümünde Osmaniye değerlendirme tarafında kalır; tetikleyici ölçümü için model sıfırdan Adıyaman+Malatya ile eğitilir); sentetik testler modelin gerçek veride öğrendiğini kanıtlamaz.

## 4. Sentetik doğrulama sonuçları (bulut, CPU, 29 Ekim 2026; gerçek veri YOK)
- `selftest_eval`: kusursuz tahmin F1=1; küçük bina + 25 px kayma → katı 0 / d=5 0 / d=15 1 (r=0); sınıf hatası, kaçırılan, yanlış alarm, geçerlilik maskesi, oylama, tetikleyici bölgeleri geçti.
- `selftest_dataset`: bölme + kilitler, kırpım–etiket hizası, 255 ignore, önbellek eşitliği, ötelemenin yalnız sonrası görüntüyü değiştirmesi, logit yolu = `ChangeOS.forward` geçti.
- 14 epoch'luk mini koşu (6 sentetik karo): kayıp düştü (c_ce 1,18→0,10; konum IoU 0,77 hold-out karoda). Hız/bellek ölçümü anlamsız (CPU).
