"""DisasterM3 (HF: Kingdrone-Junjue/DisasterM3) Türkiye dosyalarını indirir. Kaggle'da çalışacak şekilde yazıldı
(bilgisayara hiçbir şey inmez). Yeniden başlatılabilir: var olan dosyalar atlanır.

    python -m src.dm3_download --dry-run              # yalnız uzak listeyi çıkarır, sayıyı raporlar (veri inmez)
    python -m src.dm3_download --out /kaggle/working/dm3_turkiye
    python -m src.dm3_download --out /tmp/x --limit 20  # test: her kaynaktan ilk N dosya

Çıktı düzeni:  <out>/bench/<DisasterM3_Bench altı>  ve  <out>/train/<zip üyesi>
Lisans: CC BY-NC-SA 4.0 (DOI 10.48550/arXiv.2505.21089). Elde edilen dataset PUBLIC yapılmaz.
HF_TOKEN varsa (Kaggle Secrets/ortam) kullanılır; zorunlu değil.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

from src.dm3_index import tiles_from_paths

REPO = "Kingdrone-Junjue/DisasterM3"
HF = "https://huggingface.co"
RAW = f"{HF}/datasets/{REPO}/resolve/main/"
ZIPS = {"masks.zip": "DisasterM3_Instruct/masks.zip", "train_images.zip": "DisasterM3_Instruct/train_images.zip"}
BENCH = "DisasterM3_Bench/"


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def headers():
    t = hf_token()
    return {"Authorization": f"Bearer {t}"} if t else {}


def retry(f, n=5):
    err = None
    for i in range(n):
        try:
            return f()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 + 4 * i)
    raise err


def list_remote():
    """-> (bench_files [(rel, size)], {zipname: [üye adları]})  (yalnız 'turkey' içerenler)"""
    r = retry(lambda: requests.get(f"{HF}/api/datasets/{REPO}?blobs=true", headers=headers(), timeout=120))
    r.raise_for_status()
    bench = [(x["rfilename"], x.get("size") or 0) for x in r.json()["siblings"]
             if x["rfilename"].startswith(BENCH) and "turkey" in x["rfilename"].lower()]
    from remotezip import RemoteZip
    members = {}
    for zname, path in ZIPS.items():
        try:
            rz = retry(lambda: RemoteZip(RAW + path, headers=headers()))
        except TypeError:  # eski remotezip: headers parametresi yok
            rz = retry(lambda: RemoteZip(RAW + path))
        members[zname] = [i.filename for i in rz.infolist() if "turkey" in i.filename.lower() and not i.is_dir()]
    return bench, members


def dry_run():
    bench, members = list_remote()
    print(f"bench dosya: {len(bench)}  (~{sum(s for _, s in bench) / 1e9:.2f} GB)")
    for z, m in members.items():
        print(f"{z}: {len(m)} üye")
    tb = tiles_from_paths("bench", [r[len(BENCH):] for r, _ in bench])
    tt = tiles_from_paths("train", [m for m in members["masks.zip"]] + [m for m in members["train_images.zip"]])
    ok_b = sum(v["status"] == "ok" for v in tb.values()); ok_t = sum(v["status"] == "ok" for v in tt.values())
    print(f"listeden çıkan kullanılabilir karo (status==ok): bench {ok_b} + train {ok_t} = {ok_b + ok_t}")
    return bench, members


def download(out, limit=None, workers=8, no_sar=False):
    bench, members = list_remote()
    if no_sar:  # SAR (sar_post) dosyalarını atla: disk tasarrufu; öncesi/sonrası optik + maske yeter
        bench = [b for b in bench if "sar_post" not in b[0]]
        members = {z: [m for m in ms if "sar_post" not in m] for z, ms in members.items()}
    os.makedirs(out, exist_ok=True)
    log_p = os.path.join(out, "_indir.log")
    lk, done, fail = threading.Lock(), [0], []

    def log(m):
        with lk, open(log_p, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + m + "\n")
        print(m, flush=True)

    if limit:
        bench = bench[:limit]
        members = {z: m[:limit] for z, m in members.items()}

    def get_bench(item):
        rel, _ = item
        p = os.path.join(out, "bench", rel[len(BENCH):])
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)

        def one():
            with requests.get(RAW + rel, headers=headers(), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(p + ".part", "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
            os.replace(p + ".part", p)
        try:
            retry(one)
            with lk:
                done[0] += 1
        except Exception as e:  # noqa: BLE001
            fail.append((rel, str(e)[:120]))

    log(f"bench: {len(bench)} dosya")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get_bench, bench))
    log(f"bench bitti (indirilen {done[0]}, hata {len(fail)})")

    from remotezip import RemoteZip
    tl = threading.local()

    def get_member(args):
        zname, member = args
        p = os.path.join(out, "train", member)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return
        if not hasattr(tl, zname):
            try:
                setattr(tl, zname, retry(lambda: RemoteZip(RAW + ZIPS[zname], headers=headers())))
            except TypeError:
                setattr(tl, zname, retry(lambda: RemoteZip(RAW + ZIPS[zname])))
        try:
            retry(lambda: getattr(tl, zname).extract(member, os.path.join(out, "train")))
            with lk:
                done[0] += 1
        except Exception as e:  # noqa: BLE001
            fail.append((member, str(e)[:120]))

    jobs = [(z, m) for z, ms in members.items() for m in ms]
    log(f"eğitim zip üyeleri: {len(jobs)}")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get_member, jobs))
    log(f"eğitim bitti (toplam indirilen {done[0]}, hata {len(fail)})")
    if fail:
        json.dump(fail, open(os.path.join(out, "_hatalar.json"), "w"), indent=1)
        log(f"HATA: {len(fail)} dosya indirilemedi -> _hatalar.json; betiği yeniden çalıştırın (kalanları tamamlar)")
        return 1
    log("TAMAM")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/dm3_turkiye")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-sar", action="store_true", help="SAR dosyalarını indirme (disk tasarrufu)")
    a = ap.parse_args()
    if a.dry_run:
        dry_run(); sys.exit(0)
    sys.exit(download(a.out, a.limit, no_sar=a.no_sar))
