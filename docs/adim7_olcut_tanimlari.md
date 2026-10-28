# Adım 7 · Ölçüt tanımları (Makale 1) — 28 Ekim 2026

> Model sonuçlarına bakılmadan yazıldı. Sonradan değişirse neden + tarih altına eklenir. Durum: **ONAYLI (Melih, 28 Ekim 2026, Adım 8 başı)** — kademeler T0–T3/TX ve toleranslı bina düzeyi d = 15 px (duyarlılık d = 5 px) olduğu gibi kabul edildi.

## 1. Kayma kademeleri (`kayma_tier`)
Kaynak: `manifest.csv` → `kayma_post_px` (sonrası görüntünün öncesine göre otomatik kayma tahmini, piksel). Adım 6 denetim kademeleriyle (0 ≲5, 1 ≈5–15, 2 >15) uyumlu; en uç kademe ayrıca bölündü.

| Kademe | Aralık (px) | Anlamı |
|---|---|---|
| T0 | ≤ 5 | oturuyor |
| T1 | 5–15 | belirgin kayma |
| T2 | 15–30 | ağır |
| T3 | > 30 | çok ağır |
| TX | tahmin yok | ayrı raporlanır |

Tüm sonuç tabloları event satırlarına ek olarak kademeye göre kırılım içerir. Kademe, test setinde model sonuçlarına bakılarak değiştirilmez.

## 2. Üç ölçüt düzeyi
1. **Katı bina düzeyi (ana):** referans bina bileşeni (bağlı bileşen) için, tahmin maskesinin bileşen içindeki çoğunluk sınıfı; tampon 0 px. Sınıf başına F1, Macro-F1, hasar sınıfları için IoU.
2. **Toleranslı bina düzeyi:** aynı, ama tahmin, bileşenin d px genişletilmiş çevresinde aranır. **d = 15 px** (T1 sınırı; ana toleranslı ölçüt) ve duyarlılık için **d = 5 px**. Etiket gürültüsünün kayma kaynaklı kısmını ayırmak içindir.
3. **Karo düzeyi (ikincil, her koşulda raporlanır):** "karoda yıkık bina var mı", hasar yoğunluğu (hasarlı+yıkık alan oranı).

Geri dönüş tetikleyicisi (ölçüm katı, skor, eşik sayısı) 28 Ekim 2026'da, model sonuçlarına bakmadan yazıldı: bkz. `docs/adim8_tetikleyici.md`.

## 3. Değerlendirme geçerlilik maskesi
Kod: `src/tile_validity.py`. Öncesi veya sonrası görüntüsü boş (üç kanal toplamı 0) piksel değerlendirme dışı; küçük delikler kapatılır, kenar 3 px aşındırılır. Karo atılmaz, yalnız piksel maskelenir; tamamı dışarıda kalan bina bileşeni değerlendirmeye girmez. `gecerli_oran < 0.5` karolar `agir_bos=1` bayrağıyla ayrıca raporlanır (duyarlılık). Çıktı: `data_manifest/splits/tile_validity_v1.csv`.

## 4. Sızıntı kuralı
Kod: `src/leak_check.py` (ön-kayıtlı eşikler dosya başlığında). Eşleşen karo çifti: inlier ≥ 30 ve inlier/iyi eşleşme ≥ 0.30. Event çifti ≥ 3 eşleşen çiftle "ÖRTÜŞÜYOR", 1–2 "ŞÜPHELİ" (elle görsel bakış), 0 "TEMİZ". Örtüşen event, bilinen şehirle aynı tarafa alınır (`data_manifest/splits/leak_overrides_v1.csv`); iki bilinmeyen (e6–e7) örtüşürse S2'de çakışma çıkar ve Melih'e danışılır.

## 5. Bilinen asimetri (rapora yazılacak)
Eğitim şehirlerinde etiket kayması büyük (Adıyaman medyan ≈45 px, Malatya ≈34 px), test event'lerinde küçük (Antakya medyan ≈10 px, e6 ≈4 px, e7 ≈11 px). Bu, katı vs toleranslı farkı ve kademe kırılımı raporlamasını zorunlu kılar; hasar gerçekliğinden bağımsız bir alan farkı olabilir.
