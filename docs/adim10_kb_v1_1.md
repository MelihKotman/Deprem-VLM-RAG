# KB v1.1 — değişiklikler (31 Ekim 2026)

Kaynak: `data_manifest/20250622-resmi-gazete.pdf` (RG 32934 gövdesi, metin olarak okundu) ve `data_manifest/Hasar_Tespit_Genelgesi_ve_Ekleri.pdf` (20 s., sayfa 10 görsel olarak okundu). v1.0 dosyaları değişmedi (entries sha1 19a769a42b); v1.1 ayrı klasörde (`kb/kb_v1.1/`), queries_v1.jsonl aynı (sha1 1bce35f7eb), gold kimlikleri geçerli.

Değişiklikler (hepsi birincil metin doğrulamasından gelir; erişim sonucuna bakılarak yapılmadı):
1. `gen14.cracks` held → active: genelge sayfa 10 form açıklaması görsel doğrulandı (az 1–4 mm; orta 5–10 mm; ağır >10 mm, yaygın kesme kırılması; yıkık: taşıyıcı sistemde göçme).
2. `yon25.gb_te` DÜZELTİLDİ: v1.0'daki "GB/TE = girilemeyen ya da belirlenemeyen" genellemesi TE için yanlıştı. m.6/5: GB = dıştan karar verilemeyen, içeriden incelenemeyen, tespite izin verilmeyen; TE = tespit öncesi sıva/boya gibi tadilatla hasarın kapatılması. +U gerekçesinde TE "uzaktan belirlenemiyor"un resmî karşılığı olarak kullanılmaz; GB ve genelge §4 kullanılır.
3. `yon25.codes`: DD/GB/TE/KD açılımları eklendi.
4. Yeni: `yon25.def_low` (0/2/4/6 tanımları), `yon25.def_high` (AYB ve 8), `yon25.def_dd_kd` (DD ve KD; kapsam dışı yapılar m.2).
Toplam 45 kayıt (hepsi aktif).

Notlar: (a) 20 sorgu v1.1 üzerinde yeniden koşulacak; yalnız gerileme kontrolüdür, parametre ayarı yapılmaz. Tanı olarak BM25 v1.1: 17/17, MRR 0,882 (v1.0: 0,838). (b) B+ gömme bloğu büyüdü: char/4 tahmini ≈ 4.277 token (v1.0'da tahmin 3.492, ölçüm 3.849 idi → ölçüm ≈ 4,7 K beklenir); Kaggle'da Qwen3 tokenizer ile yeniden ölçülecek. (c) Yönetmelik EK-1 (taranmış PDF) bu sürümde yeniden okunmadı.
Kullanım: `KB_VER=kb_v1.1 python -m src.kb_tools validate|tokens|index|eval_hybrid`; Kaggle notebook'ta hücre 1 bunu ayarlar.
