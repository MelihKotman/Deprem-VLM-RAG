# deprem-vlm-rag

Makale 1: uydu görüntüsünden deprem hasar raporu — görsel model + VLM + sürümlü RAG (EMS-98 + hasar tespiti yönetmeliği) + LLM. Kollar A, A+, B, C, D (+U).
Özel repo. Büyük veri, makale PDF'leri ve model ağırlıkları repoda **yok** (`.gitignore`).

## Klasörler
| Klasör | İçerik |
|---|---|
| `config/` | `base.yaml`: tek yapılandırma (yollar, seed, model adı+sürüm, bölme dosyası) |
| `data_manifest/` | Küçük tablolar (manifest, bölge özeti, ileride bölme dosyaları); görüntüler Kaggle'da |
| `notebooks/` | Akış sırasıyla notebook'lar (`01_...`; sonrakiler adımı gelince) |
| `src/` | Ortak kod (config yükleyici, kontroller) |
| `kb/` | Bilgi tabanı (Adım 9–10) |
| `runs/` | Koşu çıktıları: `metrics.json`, ham model çıktıları (ağırlıklar hariç) |
| `reports/` | Tablolar, şekiller, denetim sayfaları |
| `archive/` | Yedek BRIGHT+Maxar hattı (aktif değil) |
| `docs/` | Taşıma listesi ve repo notları |

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

## Durum
Yol haritası: Adım 3–5 tamam (repo, hesaplar/anahtarlar, veri Kaggle'da: özel dataset `melihkotman/disasterm3-turkiye`). Sonraki: Adım 6 (tabakalı ~96 karo denetimi).
