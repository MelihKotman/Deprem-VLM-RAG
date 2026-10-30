# kb/ — bilgi tabanı

`kb_v1.0/entries.jsonl` (kayıtlar), `kb_v1.0/queries_v1.jsonl` (20 sorgu + gold; DONDURULMUŞ). Kayıtlar kaynaktan özet (`kaynaktan_özet`) veya bizim üretim (`bizim_üretim`); birebir alıntı yok. `durum: held` kayıtlar indekse/prompta girmez. Doğrulama ve değerlendirme: `python -m src.kb_tools validate|tokens|eval_bm25`. Karar ve ön-kayıt: `docs/adim10_kb_ve_erisim.md`. Not: EMS-98 ve yönetmelik özetleri yayımlanacaksa kaynak/lisans durumu `notlar/adim9-kaynaklar.md`'e bakılır.
