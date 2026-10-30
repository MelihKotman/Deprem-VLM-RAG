# Adım 10 — KB v1.0 ve erişim (ön-kayıt + durum), 30 Ekim 2026

## 1. Ne yapıldı
- `kb/kb_v1.0/entries.jsonl`: 42 kayıt (41 aktif, 1 `held`). Alanlar: `id, kaynak_id, kaynak_türü, sürüm, topic, text_en, term_tr, remote_observable, dogrulama, yer, durum`. Kaynak kimlikleri: `ems98, genelge_2014, yonetmelik_2025, disasterm3, aydogdu2023, map` (map = bizim_üretim eşleme/karşılaştırma). Metinler kendi sözlerimizle özet; birebir alıntı yok.
- `gen14.cracks` (genelge sayfa 10 çatlak eşikleri) `held`: OCR gürültülü, görsel doğrulama yok; indekse ve prompta GİRMEZ, doğrulanınca `active` olur ve sürüm kb_v1.1 olur.
- `yon25.codes` / `yon25.gb_te`: m.6/1 kod listesi biliniyor; DD, GB, TE, KD'nin açılımı ve m.6/5 sınıf tanımları gövde metni (Resmî Gazete sayfası otomatik araçlara kapalı) elimizde olmadığı için KB'ye girmedi; açılım yazılmadı.
- Kapanan açık kaynaklar: DisasterM3 Tablo 5/Ek A.1 okundu (yerel PDF; intact/damaged/destroyed tanımları `dm3.*` kayıtlarında). Aydoğdu Gürbüz ve Aslan (2023): sayfalar 184, 185, 189–190 teyit edildi; **dikkat: Tablo 3'te basılı TOPLAM 2.239.780, satır toplamı 2.329.780 (fark 90.000); metin s.193 "2.329.780" diyor** → toplam KB'de ve makalede kullanılmaz, yüzde (%5,9) yazılmaz. Genelge sayfa 10 görsel doğrulaması AÇIK (AFAD PDF'i robots.txt nedeniyle çekilemedi).

## 2. Erişim tasarımı (ön değerler; sorgu sonuçlarına bakılarak AYARLANMAZ)
Parça = tek kayıt (≤ 800 karakter; kayan pencere yok). İndekslenen metin: `[kaynak_id | topic] text_en (TR: term_tr)`. BM25 (k1 1,5; b 0,75) top 20 + vektör (EVREN `qwen3-embedding-8b`, kosinüs) top 20 → RRF (c = 60) → ilk 20 → rerank (`qwen3-reranker-8b`) → **K = 5**. Skor eşiği v1.0'da YOK; "kaynak yok" davranışı için eşik Adım 12 pilotunda, bu 20 sorgudan ayrı bir kümede belirlenir. Çağrı arşivi: `runs/kb/api_archive.jsonl` (model, system_fingerprint, id, created, usage, ham yanıt).

## 3. 20 sorgulu denetim — ölçüt sonuçtan ÖNCE yazıldı
Dosya: `kb/kb_v1.0/queries_v1.jsonl` (17 cevap taşıyan, 3 cevap taşımayan; her sorguda gold kayıt ve "tam/kısmen" etiketi). sha1 bu belgenin §5'inde. **Ölçüt:** cevap taşıyan 17 sorgunun en az **16'sında** (≥ 16/17) ilk 5'te gold bulunur (gold'da "tam" varsa o; yoksa "kısmen"). Cevap taşımayan 3 sorgu için yalnız rerank skorları kaydedilir (eşik yok). Sorgu ve gold dosyası ilk koşudan sonra DEĞİŞTİRİLMEZ; değişirse v2 + gerekçe.
**Sınırlılık (dürüstlük notu):** sorguları ve KB'yi aynı yazar (Claude) yazdı; sorgular KB kelimeleriyle örtüşebilir → 20 sorgu bir *duman testi*, erişim kalitesi kanıtı değil. Etiketleme uzman/bağımsız değil; makalede böyle yazılır.

## 4. Tanı: yalnız BM25 (anahtar gerekmez, 30 Ekim)
Hit@5 = 17/17, MRR 0,838. İyimser olabilir (yukarıdaki sınırlılık). Cevap taşımayan 3 sorguda da 5 sonuç döner → eşik olmadan "kaynak yok" davranışı gelmez (Adım 12). Resmî ölçüt hibrit hattın Kaggle koşusudur (`notebooks/10_kb_indeks_sorgu.ipynb`).

## 5. Dondurulan dosyalar
Bu belge yazıldığında: `entries.jsonl` sha1 `19a769a42b`, `queries_v1.jsonl` sha1 `1bce35f7eb` (kısa 10 hane; `python -m src.kb_tools validate` yazdırır).

## 6. B+ gömme bütçesi — KARAR GEREKİYOR (Melih)
Ölçüm (tahmin, karakter/4; Qwen3 tokenizer ile Kaggle'da `tokens` komutu doğrular): tam KB bloğu ≈ 3.500 token, önerilen çekirdek (EMS-98 dereceleri, dıştan inceleme kuralları, kodlar, kapsam, uzaktan gözlenebilirlik, veri seti etiketleri, eşlemeler; iç inceleme ayrıntıları, genelge, Aydoğdu, karşılaştırma HARİÇ) ≈ 2.300 token. ≈34.000 çağrıda ek yük: tam ≈ +119 M, çekirdek ≈ +78 M token (günlük 15 M havuzun ≈ 8 / 5 günü). Seçenekler: (A) B+ tam KB'yi gömer, C/D de aynı KB'yi erişir → H3 gerçekten tek değişkenli (yalnız erişim) ama pahalı ve KB küçük olduğundan erişimin kazanç sağlaması zor; (B) B+ çekirdeği gömer, C/D tam KB'yi erişir → ucuz ama H3'te içerik farkı karışır (Revizyon 3'te yazılan "aynı çekirdek" varsayımı tam doğru olmaz); (C) B+ çekirdeği gömer, C/D'nin indeksi de yalnız çekirdek → tek değişkenli, ucuz, ama erişimin işi önemsizleşir. Karar test'e bakmadan verilir.
