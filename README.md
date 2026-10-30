# deprem-vlm-rag

Makale 1: uydu görüntüsünden deprem hasar raporu — görsel model + VLM + sürümlü RAG (EMS-98 + hasar tespiti yönetmeliği) + LLM. Kollar A, B, B+, C, D (+U) (Revizyon 3: A+ yerine B+; bkz. `docs/kollar_ve_hipotezler.md`). Vurgu RQ2 (bilgi tabanı) ve RQ3 (kanıta bağlılık, "bilinmiyor").
Özel repo. Büyük veri, makale PDF'leri ve model ağırlıkları repoda **yok** (`.gitignore`).

## Klasörler
| Klasör | İçerik |
|---|---|
| `config/` | `base.yaml`: tek yapılandırma (yollar, seed, model adı+sürüm, bölme dosyası) |
| `data_manifest/` | Küçük tablolar (manifest, bölge özeti, ileride bölme dosyaları); görüntüler Kaggle'da |
| `notebooks/` | Akış sırasıyla notebook'lar (`00`–`09` hazır; sonrakiler adımı gelince, bkz. `notebooks/README.md`) |
| `src/` | Ortak kod (config yükleyici, kontroller) |
| `kb/` | Bilgi tabanı (Adım 9–10) |
| `runs/` | Koşu çıktıları: `metrics.json`, ham model çıktıları (ağırlıklar hariç) |
| `reports/` | Tablolar, şekiller, denetim sayfaları |
| `archive/` | Yedek BRIGHT+Maxar hattı (aktif değil) |
| `docs/` | Ön-kayıtlar (Adım 8, 8B, K0, K1), kol/hipotez tasarımı (`kollar_ve_hipotezler.md`), 8B kapanış özeti (`adim8b_sonuc.md`), taşıma listesi |

## Yeniden üretme (yerel, 3 komut)
```bash
git clone <repo-url> deprem-vlm-rag && cd deprem-vlm-rag
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m src.config && python -m src.check_manifest      # yapılandırma + 1.030 kullanılabilir karo doğrulaması
```
Beklenen: ikinci komut `status==ok: 1030` yazar ve çıkış kodu 0 verir.

## Kaggle'da (Adım 4–5'ten sonra)
```python
!git clone https://<token>@github.com/<kullanici>/deprem-vlm-rag.git   # token Kaggle Secrets'tan
%cd deprem-vlm-rag
!pip -q install -r requirements.txt && python -m src.config
```
API anahtarları yalnız Kaggle Secrets'ta (`EVREN_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`); yerelde `.env` (`.env.example`'dan), asla commit edilmez.

## Kurallar
- `config/base.yaml` dışında sabit yol/seed/model adı kodda tutulmaz.
- Her API çağrısında ham çıktı + `api_name` + `evren.routed_model` + tarih arşivlenir; EVREN'de `auto` yasak.
- Bölme dosyaları (`split_S1_v1.csv`, `split_S2_v1.csv`) Adım 7'de dondurulur; dondurulduktan sonra değiştirilmez, yeni sürüm açılır.
- Lisans: DisasterM3 CC BY-NC-SA 4.0, Maxar/BRIGHT CC BY-NC; repo ve Kaggle dataset'i PUBLIC yapılmaz.

## Durum (30 Ekim 2026)
Yol haritası: Adım 1–7 tamam (altyapı, veri, bölme v1 dondu); Adım 8 (ChangeOS tam eğitim + test) bitti, kalan: kademe kırılımı ve ağırlıkların saklanması; **Adım 8B kapandı** (K0, K1, Kapı T geçmedi; `docs/adim8b_sonuc.md`). **Revizyon 3:** odak RQ2/RQ3; kol tasarımında A+ yerine B+ (H3 = C vs B+). Git etiketleri: `adim8-full-v1`, `adim8-test-v1`, `adim8b-1-v1`, `adim8b-k0-v1`, `adim8b-k1-v1`. Sonraki: Adım 9 (KB kaynaklarını topla ve oku), Adım 10 (KB v1.0 ve erişim).
Yol haritası ve karar notları repoda değil, proje notlarında (`notlar/yol-haritasi-checklist.md`).
