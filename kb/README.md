# kb/ — bilgi tabanı

| Sürüm | Dosyalar | İçerik |
|---|---|---|
| `kb_v1.0/` | `entries.jsonl` (sha1 19a769a42b), `queries_v1.jsonl` (sha1 1bce35f7eb) | 42 kayıt: 41 aktif + 1 `held` (`gen14.cracks`); 20 sorgu + gold (DONDURULMUŞ) |
| `kb_v1.1/` | `entries.jsonl` (sha1 18de201b72), `queries_v1.jsonl` (aynı) | 45 kayıt, hepsi aktif; birincil metinle doğrulanmış düzeltme ve eklemeler (ayrıntı `docs/adim10_kb_v1_1.md`). v1.0 dosyalarına dokunulmadı |

- **Şema:** `id, kaynak_id, kaynak_türü, sürüm, topic, text_en, term_tr, remote_observable, dogrulama, yer, durum`. `kaynak_türü` ∈ {`kaynaktan_özet` (v1.0'da 36, v1.1'de 39), `bizim_üretim` (6)}. `durum: held` kayıtlar indekse/prompta girmez.
- **Araçlar:** `python -m src.kb_tools validate|tokens|eval_bm25|index|eval_hybrid`. Sürüm `KB_VER` ortam değişkeniyle seçilir (varsayılan `kb_v1.0`; v1.1 için `KB_VER=kb_v1.1`). Hibrit hat Kaggle T4'te yerel Qwen3-Embedding-4B + Qwen3-Reranker-4B ile çalışır; EVREN `/embeddings` ve `/rerank` uç noktaları doğrulanmadı (yedek).
- **Kullanım:** B+ kolunda KB'nin tamamı prompta gömülür, C/D aynı KB'den arar (karar 31 Ekim 2026). Karar ve ön-kayıt: `docs/adim10_kb_ve_erisim.md`.
- **Telif / yeniden kullanım:** kaynak ve lisans durumu proje notlarında (`notlar/adim9-kaynaklar.md`). 30 Ekim'deki "özet + atıf, birebir alıntı yok" kuralı 31 Ekim'de yeniden açıldı: KB belgeleri ya olduğu gibi aktarılacak ya da "LLM taslak, insan doğruladı" yoluyla hazırlanacak. Kaynak türüne göre ayırma ve `bizim_üretim` kayıtlarının KB'den çıkarılması **henüz karara bağlanmadı**; o zamana kadar `kaynak_türü` ve `dogrulama` alanları esas alınır.
