# Adım 8 · Karo düzeyine geri dönüş tetikleyicisi (ön-kayıt) — 28 Ekim 2026

> Hiçbir model sonucuna bakılmadan, hiçbir görsel model eğitilmeden yazıldı (tarih: 28 Ekim 2026). Sonradan değişirse neden + tarih altına eklenir; sayı sonuçlara bakıp değiştirilmez. Kaynak karar: `notlar/bolme-esik-karari.md` (Revizyon 1), ölçüt tanımları: `docs/adim7_olcut_tanimlari.md`.

## 1. Soru
Bina düzeyi skorun düşüklüğü model zayıflığından mı, etiket/kayma (hizalama) gürültüsünden mi? Aynı tahmin iki katılıkta değerlendirilir: katı bina düzeyi (tampon 0 px) ve toleranslı bina düzeyi (d = 15 px). Aradaki oran gürültü payını verir.

## 2. Skor ve oran (Melih onayı, 28 Ekim 2026)
- **hasar-F1** = hasarlı sınıfı F1 ile yıkık sınıfı F1'in ortalaması (bina bileşeni düzeyi, geçerlilik maskesi uygulanmış). "Sağlam" sınıfı dahil değil (baskın sınıf, kaymadan neredeyse etkilenmez). Macro-F1 ana tabloda yine raporlanır.
- **r = katı hasar-F1 / toleranslı (d = 15 px) hasar-F1.**
- Payda 0 ise (toleranslı hasar-F1 = 0) r tanımsız: tetikleyici karar vermez, model zayıflığı notu yazılır; karo düzeyi çözüm sayılmaz.

## 3. Nerede ölçülür (Melih onayı)
Yalnız S1 eğitim şehirlerinde, **Osmaniye katlaması**: Osmaniye dışarıda, eğitim Adıyaman + Malatya (şehir-bırakmalı çapraz doğrulamanın tek test benzeri katlaması). Test seti (Antakya, e6, e7) ve S2 test (Osmaniye + e7) tetikleyici ölçümünde KULLANILMAZ.

Gerekçe (veri, manifest `status=ok`): kayma ≤ 15 px karo sayısı Adıyaman 1/152 (medyan 45,5 px), Malatya 3/154 (34,1), Osmaniye 110/168 (13,6); test: Antakya 9,8, e6 3,6, e7 10,6 px. d = 15 px toleransı Adıyaman/Malatya kaymasını zaten kapatamaz; o katlamalarda r "gürültü" değil "tolerans yetmedi" ölçer ve tetikleyiciyi yapay olarak tetikler. Adıyaman ve Malatya katlamaları bilgi amaçlı raporlanır, karar vermez.

## 4. Eşik ve karar (Melih onayı)
| r (Osmaniye katlaması) | Karar |
|---|---|
| r < 0,70 | Karo düzeyi **eş-ana** (bina düzeyi de raporlu; makale ikisini birlikte öne çıkarır, "ayrı çalışma" seçeneği hocayla konuşulur) |
| 0,70 ≤ r < 0,85 | **Ara bölge**: bina düzeyi ana; toleranslı bina düzeyi ve karo düzeyi yan sonuç, sınırlılık metni |
| r ≥ 0,85 | Yalnız bina düzeyi ana; karo düzeyi ikincil (zaten her koşulda raporlanır) |

## 5. Ek varsayılanlar (Claude'un önerisi, sonuçlardan önce; itiraz yoksa geçerli)
- r nokta tahminiyle karar verilir; bootstrap %95 aralığı (blok tabanlı, karolar bitişik/örtüşük olduğundan blok = karo ızgarası komşuluğu ya da küçük küme) yalnız bilgi olarak raporlanır. Aralık iki bölgeye taşıyorsa metinde söylenir, karar değişmez.
- Ölçüm seed 42 ile yapılır (tetikleyici için tek model yeter; ana tablolardaki seed sayısından bağımsız).
- Kayma kademesine göre kırılım (T0–T3) bilgi olarak verilir; karar T0+T1 ayrımına bağlanmaz (Osmaniye'de T0 karosu yok denecek kadar az: kayma ≤ 5 px sayısı 0).
- Tetikleyici bir kez, ana S1 modelinin ilk tam eğitiminden sonra, test setine dokunulmadan hesaplanır.

## 6. Sınırlılıklar (rapora yazılacak)
- Tek katlama (Osmaniye, 168 karo, tek event) → r'nin belirsizliği büyük; bu yüzden üç bölgeli kural.
- Katlamanın eğitim etiketleri çok kaymış şehirlerden (Adıyaman, Malatya) geliyor; test benzeri hizalamada eğitilmiş modelin r'si farklı çıkabilir.
- Toleranslı ölçüt tahmini bina çevresinde arar; tek büyük hasarlı bina yığını komşu sağlam binayı örtebilir (toleranslı skor hafif iyimser).
