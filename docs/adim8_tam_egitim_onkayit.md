# Adım 8 · Tam eğitim ön-kaydı — 29 Ekim 2026

> Hiçbir tam eğitim koşusu yapılmadan, tetikleyici katlaması (Osmaniye) ve test bölmesi hiç görülmeden yazıldı (tarih: 29 Ekim 2026). Pilot bulguları (`docs/adim8_pilot_kaggle.md` §3–§9) bu kararların dayanağıdır. Sonradan değişirse neden + tarih en alta eklenir; sonuçlara bakıp değiştirilmez. Karar sohbeti: 29 Ekim 2026, Melih onayıyla.

## 1. Amaç ve kapsam
Makale 1'in birincil hipotezi H1 (B > A): ayrı görsel modelden gelen yapılandırılmış bilginin VLM raporuna katkısı. Ölçülen şey A→B farkıdır; görsel modelin kendi skoru yarışma hedefi değildir. Görsel modelin ne yapıp ne yapamadığı (bina yerelleştirme, yıkık, hasarlı) H1'in yorumu için dürüstçe raporlanır.

## 2. Sabit yapılandırma (pilot varsayılanları; tam eğitimde AYARLANMAZ)
ChangeOS (torchange 0.0.5, `ever-beta==0.5.6`), kodlayıcı r50, ImageNet ağırlığı (BRIGHT/xBD checkpoint'i yasak), 512 px kırpım, karo başı 4 kırpım, `--dam-p 0.5`, ağırlıksız CE, öteleme 0, AdamW lr 1e-4 wd 1e-2, warmup 1 epoch, fp16 (autocast + GradScaler), oylama `sqrt_inv`, **50 epoch**.
- Pilotta en çok 20 epoch denendi; 50 epoch bu yüzden ön-kayıtlı bir varsayımdır, pilot kanıtı yoktur.
- **Checkpoint kuralı:** son epoch'un ağırlığı (`model_final.pt`) kullanılır. Test'e (ya da başka bir kümeye) bakarak checkpoint/epoch seçilmez.
- Ayar yapılmaz: pilot, bu skor seviyesinde (gürültü ≈ 0,07) hiperparametreleri ayırt edemediğini gösterdi.

## 3. Koşular
| Koşu | Eğitim | Seed | Not |
|---|---|---|---|
| tetikleyici katlaması | S1 eğitim, Osmaniye dışarıda (Adıyaman+Malatya) | 42 | `--holdout Osmaniye --trigger-fold` |
| S1 ana | Adıyaman+Malatya+Osmaniye (474 karo) | 42, 43, 44 | test'e dokunulmaz |
| S2 dayanıklılık | Adıyaman+Malatya+Antakya+e6 (618 karo) | 42 (tek) | test'e dokunulmaz |

Kaba süre: tetikleyici ≈1,5 sa, S1 ≈2,3 sa/seed, S2 ≈3,0 sa (toplam ≈11,4 sa GPU). Tüm koşular birbirinden bağımsızdır (tetikleyici yalnız raporlama düzeyini belirler, eğitimi değiştirmez), bu yüzden paralel/sırasız koşulabilir.

**Teknik yeniden koşu:** yalnız teknik hata (OOM, oturum kesintisi, kod hatası) nedeniyle ve aynı yapılandırmayla. `--no-cudnn-benchmark` OOM çaresi olarak serbesttir (sonuç mantığını değiştirmez). Skor kötü çıktı diye yeniden koşu yapılmaz.

## 4. Test erişimi
S1 test (Antakya + e6 + e7) ve S2 test (Osmaniye + e7) **yalnız tüm S1 seed'leri, S2 ve tetikleyici katlaması bittikten sonra, tek turda** değerlendirilir (`--eval-only --eval-role test --allow-test`). Test sonucu bir seed için görüldükten sonra diğer seed'lerin yapılandırması değiştirilmez (zaten hepsi baştan eğitilmiş olur). Kontrol karoları (Gaziantep, hasarsız) yalnız yanlış-alarm denetimi için, aynı turda. Kuru koşuda test kullanılmaz.

## 5. Tetikleyici r — asgari skor kuralı (docs/adim8_tetikleyici.md'ye ek)
Pilotta skor gürültü tabanına yakınken katı = tol5 = tol15 çıktı ve r ≈ 1,0 mekanik olarak "yalnız bina" verdi; ön-kayıtta "sıfıra yakın" için kural yoktu.
- **Kural:** Osmaniye katlamasında toleranslı (d = 15 px) hasar-F1 **< 0,15** ise r **"bilgi vermiyor"** sayılır (`bilgi_vermiyor`). Tetikleyici karar vermez.
- **Gerekçe:** 0,15 ≈ 2 × pilot gürültüsü (≈ 0,07, Malatya hold-out, 105 hasarlı + 76 yıkık bina).
- **Bilinen sınır:** Osmaniye'de hasarlı + yıkık bina Malatya hold-out'tan az (eğitim karolarında 41 + 63), gürültü orada büyük olasılıkla ≥ 0,07'dir; 0,15 bu yüzden gevşek bir eşiktir ve katlamanın "bilgi vermiyor" çıkması beklenen bir sonuçtur.
- **Karar tablosu:** tol15 hasar-F1 ≥ 0,15 ise `docs/adim8_tetikleyici.md` §4 aynen geçerli (r < 0,70 karo eş-ana; 0,70–0,85 ara; ≥ 0,85 yalnız bina). `bilgi_vermiyor` ise **bina düzeyi ana, karo düzeyi ana tabloda yan yana ama ikincil**; bu "karo düzeyi eş-ana" kararı vermez. Sınırlılık metnine yazılır; konu hocayla bir sonraki görüşmede açılır.
- Kod: `src/dm3_eval.trigger_decision`, `config/base.yaml → evaluation.fallback_trigger.min_tol15_damage_f1`, birim testi `selftest_eval` §8.

## 6. Raporlama (test sonuçlarından önce sabitlendi)
- Ana tabloda **sınıf bazlı F1** (sağlam / hasarlı / yıkık) + hasar-F1 + Macro-F1; **karo bazlı bootstrap %95 aralığı** (2.000 yeniden örnekleme); katı, tol5 ve tol15 yan yana; kayma kademesi (T0–T3) kırılımı. Event içi karo örtüşmesi için blok/küme ayrıntısı Adım 14'te netleşir; daha kaba küme yalnız aralığı genişletebilir.
- **İkincil ölçüt:** ikili "hasar var/yok" (hasarlı + yıkık birleşik). Pilot hold-out bulgusundan doğdu; test'ten bağımsız, test'e bakmadan kaydedildi.
- Her seed ayrı satırda, ayrıca seed ortalaması ± dağılım. S1 birincil, S2 dayanıklılık (bağımsız kanıt sayılmaz).
- Şehir/event bazlı tablo; bilinmeyen kaynaklı event 6 ve 7 ayrı satır.
- **Dürüstlük kuralları:** Malatya hold-out skorları "ayarlama vekili" olarak yazılır, test tahmini diye sunulmaz. Eğitim-içi skor (≈ 0,32) hiçbir yerde genelleme diye gösterilmez. Hasarlı sınıfındaki zayıflık (eğitim içinde bile ≈ 0,11) bulgu ve sınırlılık olarak yazılır, saklanmaz. Antakya sonucu Malatya'dan farklı çıkabilir; bunu önceden varsaymıyoruz.

## 7. Adım 11'e devreden
B kolunda VLM'e hangi hasar düzeyinin (üç sınıf mı, ikili mi, belirsizlik işaretli mi) verileceği Adım 11'de, test sonuçlarına bakmadan, tetikleyici katlaması ve pilot/ayarlama sonuçlarına göre kararlaştırılır.

## 8. Altyapı ve kayıt
Kaggle, **Commit (Save & Run All)** modu: koşu Kaggle sunucusunda sürer, bilgisayar/internet kapansa devam eder. Kod, git etiketine sabitlenir: `adim8-full-v1` (not defterinin çıktısındaki `code_version.txt` git sha + src/config özeti + paket sürümlerini taşır). Kaggle kotası 29 Ekim 2026'da 21 sa 51 dk kalmıştı. Colab Pro şimdilik gerekmiyor.

## 9. Değişiklik kaydı
- 29 Ekim 2026: belge oluşturuldu.
- 30 Ekim 2026 (tam eğitimler BİTTİ; test'e HİÇ bakılmadan; tetikleyici katlaması ve eğitim-içi sanity sonuçları görüldü): test öncesi **yalnız değerlendirme/raporlama kodu eklendi**, eğitim ağırlıkları ve yapılandırma değişmedi. (1) `src/dm3_eval.Acc` artık `per_tile.csv`'ye karo başına bileşen sayaçlarını (karışım + yanlış alarm; katı/tol5/tol15) yazıyor; çünkü §6'daki karo bazlı bootstrap ve ikili "hasar var/yok" ölçütü mevcut `per_tile.csv` ile hesaplanamıyordu (yalnız karo düzeyi yoğunluk satırları vardı). (2) `src/test_report.py`: sınıf bazlı F1, hasar-F1, Macro-F1, ikili hasar var/yok F1; karo bazlı bootstrap %95 AA (B=2000, seed 42, yüzdelik); düzeyler katı/tol5/tol15; kırılımlar ALL/group/tier; birim testi `src/selftest_report.py` geçti. (3) Test not defteri `notebooks/06_test_degerlendirme.ipynb`: tek tur, kapı (4 checkpoint + tetikleyici sonucu girdi olarak ekli değilse çalışmaz), kod etiketi `adim8-test-v1` (eğitim etiketi `adim8-full-v1` kalır).
