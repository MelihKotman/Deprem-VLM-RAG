# Adım 8B / K1 — gölge denemesi (ön-kayıt, 30 Ekim 2026; kod ve sonuçtan ÖNCE yazıldı)

Bağlam: K0 (kalıntı ofset) H1'i düşürdü (`adim8b-k0-sonuc.md`). Hoca geometri/yükseklik-kaybı fikrini önerdi; literatürde yükseklik değişimi ya stereo/DEM (Tong ve ark. 2012) ya bilinen bina yüksekliğiyle gölge (Tong ve ark. 2013) ile kullanılıyor. Bizde ikisi yok. K1, tek sonrası görüntüden bilgi çıkaran, K0'dan FARKLI bir ölçüdür: yıkılan binanın gölgesi kısalır/zayıflar. Tek bir kapılı deneme; geçmezse geometri kolu KAPANIR (makalede kısa negatif bulgu / gelecek çalışma).

## Hipotez (H1-gölge)
Aynı şehirde, aynı güneş geometrisinde, bina alanı sabitken yıkık/hasarlı bina ayakta binaya göre daha zayıf/kısa gölge bırakır.

## Veri
S1 eğitim şehirleri Adıyaman ve Malatya, TÜM karolar (`split_S1_v1`, role=train). Test event'leri (Antakya, e6, e7), Osmaniye ve Gaziantep kullanılmaz. Sonrası görüntü K0 ile aynı: şehir medyan kaymasıyla hizalanmış (`read_tile(shift=…)`). Bina = sınıf maskesinin 8-bağlantılı bileşeni, alan ≥ 40 px, geçerlilik ≥ %90 (K0 ile aynı).

## Ölçü (hepsi sabit)
- Gri = RGB→gri. Zemin maskesi OK = geçerli ∧ (hiçbir bina maskesinin 7×7 genişletilmişi içinde değil).
- 16 yön (22,5° adım; birim vektör (cos φ, sin φ), x sağ, y aşağı). Bina için yön k'da, t ∈ {4,8,…,32} px: bölge_t = bina maskesinin t·u kadar ötelenmişi ∧ OK; ≥ 30 piksel yoksa t atlanır. Yerel zemin parlaklığı bg = bina çevresinde (bbox ± 36 px) OK piksellerin 75. yüzdeliği (≥ 200 piksel; medyan, selftest'te binanın kendi gölgesinden etkilendiği için GERÇEK VERİ KOŞULMADAN önce değiştirildi). C_t = 1 − ortalama_gri(bölge_t)/bg. S_k = C_t'lerin ortalaması (≥ 4 geçerli t gerekir; yoksa bina atılır).
- **Gölge yönü, şehir başına, sınıf etiketi KULLANILMADAN:** tüm binaların ortalama S_k'sı; yön* = argmax_k. Keskinlik = max − medyan (16 yön). Keskinlik < 0,01 → ölçüm "ölçülemez" (ne geçti ne kaldı, raporlanır).
- Skor = −S_{yön*}; pozitif = hasarlı ∪ yıkık.
- Yükseklik karıştırıcısı: bina alanı tertilleri (şehir içinde, tüm binalarla) → **tabakalı AUC** = tertil içi AUC'lerin n_poz·n_neg ağırlıklı ortalaması.

## KAPI K1
Her iki şehirde de: tabakalı AUC ≥ 0,60 VE karo-bootstrap (1.000) %95 AA alt sınırı > 0,55. Biri bile sağlamazsa → geometri kolu KAPANIR. Geçerse → ikinci aşamaya girdi olarak (iii) kolu ön-kaydı yazılır (ayrı adım).

## Betimleyici (karar değil)
Yıkık-yalnız vs sağlam AUC; işaretsiz AUC; güneş-karşı-yön kontrolü (yön*+180°, AUC ≈ 0,5 beklenir); sağlam binalarda alan–S Spearman (pozitif beklenir: büyük bina daha uzun gölge); tertil AUC'leri.

## Sınırlılıklar
Şehir başına tek gölge yönü (birden çok sahne tarihi olabilir); komşu bina gölgesi/yoğun blok; enkaz gölgesi; güneş yüksekliği bilinmiyor (bina yüksekliği ölçülmez, yalnız gölge zayıflığı); ChangeOS bu bilgiyi örtük öğrenmiş olabilir (K1 geçse bile ek katkı ayrıca ölçülmeli); tek yapay zekâ analisti.
