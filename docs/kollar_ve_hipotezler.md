# Kollar ve hipotezler (Revizyon 3, 30 Ekim 2026)

Kayıt tarihi: 30 Ekim 2026, VLM/LLM kolları koşulmadan ÖNCE. Bu dosya, önceki "A, A+, B, C, D (+U)" tanımını ve ilk hipotez kademelerini (27 Ekim) değiştirir. Ayrıntılı gerekçe proje notlarında: `notlar/revizyon-3-rq-odagi.md`.

## Neden değişti
- A+ (VLM + prompta gömülü bilgi) görsel girdi içermiyordu; H3 (C vs A+) bu yüzden iki şeyi birden değiştiriyordu (görsel girdi + erişim).
- Yeni tasarımda **B+ = B + prompta gömülü bilgi**; H3 = **C vs B+** tek değişkenlidir (yalnız erişim). C ve D, B temelinde kalır.
- Görsel model (ChangeOS) hasar sınıflarında zayıf çıktı (S1 test hasar-F1 0,101 ± 0,037; 8B kapandı). Bu yüzden vurgu RQ2 (bilgi tabanı) ve RQ3 (kanıta bağlılık, "bilinmiyor") üzerine kaydı; RQ1 ikincil.

## Kollar
| Kol | Görsel girdi | Bilgi | Ne ayrıştırır |
|---|---|---|---|
| A | yok | yok | temel çizgi (VLM tek başına) |
| B | ChangeOS çıktısı | yok | görsel girdinin katkısı (RQ1, ikincil) |
| B+ | ChangeOS çıktısı | prompta gömülü (erişimsiz) | gömme etkisi |
| C | ChangeOS çıktısı | erişim (vektör + BM25 + rerank) | C vs B+ = erişim mi, yalnız bilgi mi (H3) |
| D | ChangeOS çıktısı | hibrit erişim | D vs C (H4) |
| +U | (eklenir) | | belirsizlik / "bilinmiyor" çıktısı |

İsteğe bağlı ikincil tanı: kâhin-görsel kolu (doğru etiketten türetilmiş görsel girdi; üst sınır; sızıntı uyarısıyla; birincil değil).

## B kolu girdi kuralı (B, B+, C, D için)
ChangeOS'tan yalnız bina yerelleştirme/sayı ve "yıkık var" pozitifleri kanıt olarak verilir (karo düzeyi kesinlik ≈ 0,9). Negatif çıktı "bilinmiyor / kanıt yok" olarak geçer, "sağlam" olarak asla: model hasarlı binayı güvenle sağlam görüyor (8B-1). Kanıt hiyerarşisi: görsel > getirilen metin > ön bilgi. Girdi düzeyinin ayrıntısı Adım 11'de, test'e bakmadan.

## Hipotez kademeleri
| Kademe | Hipotez | Çözümleme |
|---|---|---|
| Birincil | H2: C > B (cevap taşıyan alt kümede) | Önceden kayıtlı, Holm |
| Birincil | H3: C > B+ | Önceden kayıtlı |
| Birincil | H4: D ≥ C | Önceden kayıtlı |
| Birincil | RQ3: kanıtsız iddia (3 düzey), "bilinmiyor" davranışı, risk–kapsama | Önceden kayıtlı (Adım 14'te ayrıntı) |
| İkincil | H1: B > A (RQ1); B → B+; H1a (ara dereceler); H5 (+U) | Betimleyici / risk–kapsama |
| Keşifsel | kâhin-görsel kolu; kalan ablation'lar | Ek |

## KB sonucu (Adım 10) — 31 Ekim 2026'da sonuçlandı, bkz. en alttaki bölüm
B+ bilgiyi gömer, C/D erişir. Gömülecek KB sürümü token olarak sayılır; sığmazsa gömülü kısım = çekirdek tablo, kural test'e bakmadan yazılır; B+ ve C aynı çekirdeği görür.

## Yapılandırma
`config/base.yaml` kol listesi taşımıyor; kollar Adım 11'de istem dosyalarıyla (sürümlü) tanımlanır.

## Güncelleme (31 Ekim 2026) — B+ gömme bütçesi kararı
Kollar, kol tasarımı ve hipotez kademeleri DEĞİŞMEDİ; yalnız yukarıdaki koşullu kural ("sığmazsa çekirdek") sonuçlandı.
- Qwen3 tokenizer ile ölçüm (KB v1.0): tam KB (41 aktif kayıt) = 3.849 token; çekirdek (30 kayıt) = 2.488 token. KB v1.1 (45 kayıt) için beklenti ≈ 4,7 K token; Kaggle'da yeniden ölçülecek.
- **Karar (Melih, 31 Ekim 2026): seçenek A.** B+ prompt'unda tam KB gömülür; C/D aynı KB'den arar. Çekirdek tablosu elendi. Gerekçe: çekirdek seçimi bizim kararımız olur ve hakem itirazına açıktır; erişim tek değişken kalır.
- **Açık:** KB tamamı prompta sığdığı için B+ ≥ C çıkabilir. H3'ün çerçevesi ("erişim küçük KB'de de kazandırır" mı, yoksa "erişim KB prompta sığmayacak kadar büyüdüğünde gerekir" mi) hocalarla konuşulacak; ikincisiyse KB büyütülüp ikinci ölçüm gündeme gelir. H3 kaydı (C > B+) bu karardan etkilenmez, yorumlanışı etkilenir.
- KB dili: İngilizce (istem, rapor, hakem, indeksli KB metni); Türkçe resmî terimler `term_tr`'de (Adım 9 kararı, 30 Ekim).
