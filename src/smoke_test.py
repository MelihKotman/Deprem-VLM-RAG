"""Adım 4 duman testi (Kaggle'da veya yerelde). Yalnız standart kütüphane + (varsa) PIL.
Kullanım:
    python -m src.smoke_test gpu        # nvidia-smi
    python -m src.smoke_test secrets    # anahtarlar var mı (değerleri YAZMAZ)
    python -m src.smoke_test evren      # metin çağrısı + routed_model
    python -m src.smoke_test evren_img  # görüntü boyut sınırı (512..4096 px) + seed tekrarlanabilirliği
    python -m src.smoke_test gemini     # 1 metin çağrısı
    python -m src.smoke_test hf         # HF token geçerli mi (whoami)
    python -m src.smoke_test gemini_models  # anahtarın görebildiği modellerin API adları (GEMINI_MODEL için)
Ortam değişkenleri / Kaggle Secrets: EVREN_API_KEY, EVREN_BASE_URL, EVREN_MODEL (varsayılan qwen3-vl-30b),
GEMINI_API_KEY, GEMINI_MODEL, HF_TOKEN. Anahtar değerleri asla yazdırılmaz.
Yalnız sentetik görüntü gönderilir (kişisel veri yok).
"""
import base64, io, json, os, subprocess, sys, time, urllib.request, urllib.error


def secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def post(url, headers, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400], time.time() - t


def gpu():
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "nvidia-smi YOK / GPU kapalı")
    print(subprocess.run(["df", "-h", "/kaggle/working", "/tmp"], capture_output=True, text=True).stdout)


def secrets():
    for k in ("EVREN_API_KEY", "EVREN_BASE_URL", "GEMINI_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
        print(f"{k:16s}", "VAR" if secret(k) else "YOK")


def _evren(messages, **kw):
    base, key = secret("EVREN_BASE_URL"), secret("EVREN_API_KEY")
    assert base and key, "EVREN_BASE_URL / EVREN_API_KEY eksik"
    body = {"model": secret("EVREN_MODEL") or "qwen3-vl-30b", "messages": messages, "max_tokens": 2048, **kw}
    return post(base.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"}, body)


def _summ(code, resp, dt):
    if code != 200:
        return f"HTTP {code} ({dt:.1f}s): {resp}"
    u = resp.get("usage", {})
    return f"200 ({dt:.1f}s) prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} routed={resp.get('evren', {}).get('routed_model')}"


def evren():
    c, r, dt = _evren([{"role": "user", "content": "Reply with the single word: ok"}], temperature=0)
    print(_summ(c, r, dt))


def _img(n):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (n, n), (90, 120, 90)); d = ImageDraw.Draw(im)
    for i in range(0, n, max(n // 8, 1)):
        d.rectangle([i, i, i + n // 16, i + n // 16], fill=(200, 60, 60))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), len(b.getvalue())


def evren_img():
    for n in (512, 1024, 2048, 4096):
        uri, size = _img(n)
        msg = [{"role": "user", "content": [{"type": "text", "text": "Count the red squares roughly. One short sentence."},
                                            {"type": "image_url", "image_url": {"url": uri}}]}]
        print(f"{n}px / {size//1024} KB ->", _summ(*_evren(msg, temperature=0)))
    print("--- seed testi (aynı istek 2x, temperature=0.7, seed=42) ---")
    outs = []
    for _ in range(2):
        c, r, dt = _evren([{"role": "user", "content": "Write one random sentence about a river."}], temperature=0.7, seed=42)
        print(_summ(c, r, dt))
        outs.append(r["choices"][0]["message"]["content"] if c == 200 else None)
    print("seed kabul + aynı çıktı:", outs[0] is not None and outs[0] == outs[1])


def gemini():
    key, model = secret("GEMINI_API_KEY"), secret("GEMINI_MODEL")
    assert key and model, "GEMINI_API_KEY / GEMINI_MODEL (AI Studio'daki tam model adı) eksik"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    c, r, dt = post(url, {"x-goog-api-key": key}, {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]})
    print(f"HTTP {c} ({dt:.1f}s)", (r.get("usageMetadata") if c == 200 else r))


def gemini_models():
    key = secret("GEMINI_API_KEY"); assert key, "GEMINI_API_KEY eksik"
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        for m in json.loads(r.read()).get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].replace("models/", ""), "|", m.get("displayName"))


def hf():
    tok = secret("HF_TOKEN"); assert tok, "HF_TOKEN eksik"
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read()); print("HF kullanıcı:", j.get("name"), "| token türü:", j.get("auth", {}).get("accessToken", {}).get("role"))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
