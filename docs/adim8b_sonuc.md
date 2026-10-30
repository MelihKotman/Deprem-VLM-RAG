# Adım 8B — kapanış özeti (30 Ekim 2026)

Ön-kayıt: `docs/adim8b_onkayit.md` (sonuçlardan önce yazıldı); K0/K1 ayrıntıları: `docs/adim8b_k0.md`, `docs/adim8b_k1.md`. Kapılar sonuçlara bakmadan sabitti ve sonradan gevşetilmedi. Tüm 8B sonuçları test sonrası tasarlanan, keşfedici (post hoc) etiketlidir; S1 test event'leri (Antakya, e6, e7) 8B'de açılmadı.

| Kapı | Ölçüt | Sonuç | Karar |
|---|---|---|---|
| K0 kalıntı ofset (Adıyaman / Malatya) | iki şehirde AUC ≥ 0,55 | 0,542 / 0,508 | GEÇMEDİ (Kapı E koşulmadı) |
| K1 gölge (Adıyaman / Malatya) | tabakalı AUC ≥ 0,60 ve CI alt sınırı > 0,55, iki şehirde | 0,555 / 0,606 | GEÇMEDİ (Adıyaman) |
| Kapı T karar­sız bant | hasarlı∪yıkık binaların ≥ %20'si p ∈ [0,3; 0,7] | %1,3 | GEÇMEDİ |

Sonuç: 8B kapandı; kol B ChangeOS çıktısıyla ilerler. Geometri ve karo-çevirme girdisi eklenmez.

8B-1 (kalibrasyon), şehir-bırakmalı, seed 42: model hasarlı binayı kararsız değil, güvenle "sağlam" görüyor; kalibre + F1-optimum eşik ikili hasar F1'ini 2/3 şehirde artırdı (+0,030 / +0,065 / −0,034) ama mutlak F1 ≤ 0,11 ve şehir aktarımı kırılgan → en fazla yardımcı varyant. K1'de yıkık-yalnız AUC 0,64 / 0,69 betimleyici bir sinyal; K0 ölçümü gürültülü olduğundan "geometri hiçbir bilgi taşımaz" kanıtı değildir.

Makalede: 8B ana metinde yer almaz (Revizyon 3); ek tablo olabilir. Ayrıntılar: proje notları `adim8b-k0-sonuc.md`, `adim8b-k1-sonuc.md`, `adim8b1-sonuc.md`. Makale 3 adayı (yükseklik/stereo verisi) ertelendi.

Çıktılar (repo dışı): `ciktilar/adim-8b_k0/`, `ciktilar/adim-8b-k1/`, `ciktilar/adim-8b1/` (kalibrasyon: `analiz/`).
