"""Adım 10: KB v1.0 araçları — doğrulama, BM25, EVREN gömme/rerank, hibrit erişim, 20 sorgu değerlendirmesi.
Kullanım (repo kökünden):
    python -m src.kb_tools validate                 # şema + yinelenen id + yasak desen denetimi
    python -m src.kb_tools tokens [--tokenizer HF_ID]  # gömülecek blok token sayısı (tokenizer yoksa tahmin)
    python -m src.kb_tools eval_bm25                # yalnız BM25 (anahtar gerekmez), tanı amaçlı
    python -m src.kb_tools probe_embed | probe_rerank   # EVREN uç noktası var mı (Kaggle)
    python -m src.kb_tools index                    # gömmeleri hesapla -> runs/kb/kb_v1.0_embeddings.npy
    python -m src.kb_tools eval_hybrid              # BM25 + vektör + RRF + rerank, 20 sorgu, ön-kayıtlı ölçüt
KB_BACKEND=local (varsayılan, Kaggle GPU, açık ağırlık) | evren. Ön-kayıt: docs/adim10_kb_ve_erisim.md. Yalnız kamuya açık metin gönderilir. Anahtar değerleri yazdırılmaz."""
import json, math, os, re, sys, time, hashlib, collections
from pathlib import Path
KB = Path(__file__).resolve().parent.parent / "kb"
VER = "kb_v1.0"
REQ = ["id", "kaynak_id", "kaynak_türü", "sürüm", "topic", "text_en", "term_tr", "remote_observable", "dogrulama", "yer", "durum"]
SRC = {"ems98", "genelge_2014", "yonetmelik_2025", "disasterm3", "aydogdu2023", "map"}
TUR = {"kaynaktan_özet", "bizim_üretim"}
CFG = dict(bm25_k=20, dense_k=20, rrf_c=60, rerank_in=20, final_k=5, embed_model="qwen3-embedding-8b", rerank_model="qwen3-reranker-8b",
           local_embed="Qwen/Qwen3-Embedding-4B", local_rerank="Qwen/Qwen3-Reranker-4B")
BACKEND = os.environ.get("KB_BACKEND", "local")   # local (Kaggle GPU, açık ağırlık) | evren (API)
TASK = "Given a question about earthquake building damage assessment, retrieve the knowledge entries that answer it"

def load(active_only=True):
    es = [json.loads(l) for l in open(KB / VER / "entries.jsonl", encoding="utf-8") if l.strip()]
    return [e for e in es if e["durum"] == "active"] if active_only else es

def queries():
    return [json.loads(l) for l in open(KB / VER / "queries_v1.jsonl", encoding="utf-8") if l.strip()]

def embed_text(e):  # indekslenen ve (B+ için) prompta gömülen metin
    return f"[{e['kaynak_id']} | {e['topic']}] {e['text_en']} (TR: {e['term_tr']})"

def sha(path): return hashlib.sha1(open(path, "rb").read()).hexdigest()[:10]

def validate():
    es = load(False); ids = set(); bad = []
    for e in es:
        for k in REQ:
            if k not in e: bad.append((e.get("id"), "eksik alan " + k))
        if e["id"] in ids: bad.append((e["id"], "yinelenen id"))
        ids.add(e["id"])
        if e["kaynak_id"] not in SRC: bad.append((e["id"], "bilinmeyen kaynak_id"))
        if e["kaynak_türü"] not in TUR: bad.append((e["id"], "kaynak_türü"))
        if e["durum"] not in ("active", "held"): bad.append((e["id"], "durum"))
        if len(e["text_en"]) > 800: bad.append((e["id"], "metin çok uzun (alıntı/telif riski)"))
        if '"' in e["text_en"] and e["kaynak_türü"] == "kaynaktan_özet": bad.append((e["id"], "tırnak: birebir alıntı olabilir"))
    for q in queries():
        for g in q["gold"]:
            if g not in ids: bad.append((q["qid"], "gold kaydı yok: " + g))
    act = load(); print(f"kayıt {len(es)} (aktif {len(act)}, held {len(es)-len(act)}); sorgu {len(queries())}; sha1 entries={sha(KB/VER/'entries.jsonl')} queries={sha(KB/VER/'queries_v1.jsonl')}")
    print("SONUÇ:", "OK" if not bad else bad); return not bad

CORE_TOPICS = {"scale_principle", "grade_definition", "remote_observability", "damage_codes", "scope_principle", "external_inspection_rule", "imagery_rule", "unknown_outcome", "label_definition", "class_mapping"}
def is_core(e):  # B+ için ÖNERİLEN çekirdek (kural içerik/test'e bakmadan konu listesiyle; yalnız Melih onayından sonra kesinleşir)
    return e["topic"] in CORE_TOPICS and e["kaynak_id"] in {"ems98", "yonetmelik_2025", "disasterm3", "map"} and e["id"] != "dm3.label_note"

def tokens(tok_id=None, core=False):
    block = "\n".join(embed_text(e) for e in load() if (is_core(e) or not core))
    if tok_id:
        from transformers import AutoTokenizer
        n = len(AutoTokenizer.from_pretrained(tok_id).encode(block)); how = tok_id
    else:
        n = round(len(block) / 4.0); how = "TAHMİN (karakter/4); Qwen3 tokenizer ile Kaggle'da ölç"
    print(f"{'ÇEKİRDEK' if core else 'TAM KB'} gömülecek blok: {len(block)} karakter, ≈{n} token [{how}]"); return n

# ---- BM25 (bağımlılıksız) ----
def tok(s): return re.findall(r"[a-z0-9ğüşıöç]+", s.lower())
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.d = [tok(x) for x in docs]; self.N = len(docs); self.avg = sum(map(len, self.d)) / self.N; self.k1, self.b = k1, b
        df = collections.Counter(t for d in self.d for t in set(d)); self.idf = {t: math.log(1 + (self.N - n + .5) / (n + .5)) for t, n in df.items()}
    def score(self, q):
        qs = tok(q); out = []
        for d in self.d:
            tf = collections.Counter(d); s = 0.0
            for t in qs:
                if t in tf: s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out

def rank(scores, k): return [i for i, s in sorted(enumerate(scores), key=lambda x: -x[1])[:k] if s > 0]
def rrf(lists, c=60):
    sc = collections.defaultdict(float)
    for L in lists:
        for r, i in enumerate(L): sc[i] += 1.0 / (c + r + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]

# ---- EVREN ----
def secret(n):
    v = os.environ.get(n)
    if v: return v
    try:
        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret(n)
    except Exception: return None

def _post(path, body):
    import urllib.request, urllib.error
    base, key = secret("EVREN_BASE_URL"), secret("EVREN_API_KEY"); assert base and key, "EVREN_BASE_URL / EVREN_API_KEY eksik"
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r: resp = json.loads(r.read()); code = r.status
    except urllib.error.HTTPError as e: resp, code = e.read().decode(errors="replace")[:500], e.code
    rec = dict(path=path, code=code, latency_s=round(time.time() - t, 2), date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               model=resp.get("model") if isinstance(resp, dict) else None, system_fingerprint=resp.get("system_fingerprint") if isinstance(resp, dict) else None,
               id=resp.get("id") if isinstance(resp, dict) else None, created=resp.get("created") if isinstance(resp, dict) else None,
               usage=resp.get("usage") if isinstance(resp, dict) else None)
    out = Path(os.environ.get("RAG_RUNS", "runs")) / "kb"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "api_archive.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(dict(meta=rec, request_hash=hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10], raw=resp), ensure_ascii=False) + "\n")
    return code, resp, rec

_M = {}
def _local_embedder():
    if "e" not in _M:
        from sentence_transformers import SentenceTransformer
        import torch
        _M["e"] = SentenceTransformer(CFG["local_embed"], model_kwargs={"torch_dtype": torch.float16}, device="cuda" if torch.cuda.is_available() else "cpu")
    return _M["e"]

def _local_rerank(query, docs):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if "r" not in _M:
        tk = AutoTokenizer.from_pretrained(CFG["local_rerank"], padding_side="left")
        md = AutoModelForCausalLM.from_pretrained(CFG["local_rerank"], torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        pre = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        _M["r"] = (tk, md, tk.convert_tokens_to_ids("no"), tk.convert_tokens_to_ids("yes"), pre, suf)
    tk, md, no, yes, pre, suf = _M["r"]; out = []
    for d in docs:
        txt = pre + f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {d}" + suf
        x = tk([txt], return_tensors="pt").to(md.device)
        with torch.no_grad(): lg = md(**x).logits[0, -1, [no, yes]].float()
        out.append(torch.softmax(lg, 0)[1].item())
    return out

def embed(texts, is_query=False):
    if BACKEND == "local":
        import numpy as np
        v = _local_embedder().encode(texts, prompt=(f"Instruct: {TASK}\nQuery:" if is_query else None), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(v, dtype="float32"), dict(backend="local", model=CFG["local_embed"])
    c, r, m = _post("/embeddings", {"model": CFG["embed_model"], "input": texts}); assert c == 200, (c, r)
    import numpy as np; return np.array([d["embedding"] for d in r["data"]], dtype="float32"), m

def rerank(query, docs, top_n):
    if BACKEND == "local":
        sc = _local_rerank(query, docs); order = sorted(range(len(sc)), key=lambda i: -sc[i])[:top_n]
        return [(i, sc[i]) for i in order], dict(backend="local", model=CFG["local_rerank"])
    c, r, m = _post("/rerank", {"model": CFG["rerank_model"], "query": query, "documents": docs, "top_n": top_n}); assert c == 200, (c, r)
    res = r.get("results") or r.get("data"); return [(x["index"], x.get("relevance_score", x.get("score"))) for x in res], m

def probe_embed():
    if BACKEND == "local":
        v, m = embed(["probe"]); print("local embed", v.shape, m); return
    c, r, m = _post("/embeddings", {"model": CFG["embed_model"], "input": ["probe"]}); print(c, m, (len(r["data"][0]["embedding"]) if c == 200 else r))
def probe_rerank():
    if BACKEND == "local":
        print("local rerank", rerank("fine cracks", ["hair-line cracks in plaster", "total collapse of the building"], 2)); return
    c, r, m = _post("/rerank", {"model": CFG["rerank_model"], "query": "probe", "documents": ["a", "b"], "top_n": 2}); print(c, m, r if c != 200 else list(r)[:5])

def index():
    import numpy as np
    es = load(); V, m = embed([embed_text(e) for e in es]); Path(os.environ.get("RAG_RUNS", "runs"), "kb").mkdir(parents=True, exist_ok=True)
    np.save(Path(os.environ.get("RAG_RUNS", "runs")) / "kb" / f"{VER}_embeddings.npy", V); print("gömme", V.shape, m)

# ---- değerlendirme ----
def _score(ranked_ids, q):
    g = q["gold"]; tam = [i for i, v in g.items() if v == "tam"]; need = tam or list(g)
    hit = any(i in ranked_ids for i in need)
    rr = next((1 / (r + 1) for r, i in enumerate(ranked_ids) if i in need), 0.0)
    return hit, rr

def evaluate(retr, name):
    es = load(); qs = queries(); hits = 0; mrr = 0; rows = []
    for q in qs:
        ids = [es[i]["id"] for i in retr(q["query"], es)[:CFG["final_k"]]]
        if q["answer_bearing"]:
            h, rr = _score(ids, q); hits += h; mrr += rr; rows.append((q["qid"], "VAR", "hit" if h else "MISS", ids))
        else: rows.append((q["qid"], "YOK", "-", ids))
    nb = sum(q["answer_bearing"] for q in qs)
    for r in rows: print(*r)
    print(f"[{name}] Hit@{CFG['final_k']} (cevap taşıyan) {hits}/{nb}; MRR {mrr/nb:.3f}; ön-kayıtlı ölçüt: ≥ 16/{nb} -> {'GEÇTİ' if hits >= 16 else 'GEÇMEDİ'}")

def eval_bm25():
    es = load(); bm = BM25([embed_text(e) for e in es]); evaluate(lambda q, es_: rank(bm.score(q), CFG["bm25_k"]), "BM25")

def eval_hybrid():
    import numpy as np
    es = load(); bm = BM25([embed_text(e) for e in es]); V = np.load(Path(os.environ.get("RAG_RUNS", "runs")) / "kb" / f"{VER}_embeddings.npy")
    V = V / np.linalg.norm(V, axis=1, keepdims=True); scores_log = {}
    # T4 (≈15 GB) iki 4B fp16 modeli birlikte taşımaz: önce tüm sorgu vektörleri, sonra gömücü bellekten atılır, sonra rerank.
    QV = {q["query"]: embed([q["query"]], is_query=True)[0][0] for q in queries()}
    if "e" in _M:
        import gc, torch; del _M["e"]; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    def retr(q, es_):
        qv = QV[q] / np.linalg.norm(QV[q])
        dense = rank((V @ qv).tolist(), CFG["dense_k"]); lex = rank(bm.score(q), CFG["bm25_k"])
        fused = rrf([lex, dense], CFG["rrf_c"])[:CFG["rerank_in"]]
        rr, _ = rerank(q, [embed_text(es[i]) for i in fused], CFG["final_k"]); scores_log[q] = [(es[fused[i]]["id"], s) for i, s in rr]
        return [fused[i] for i, _ in rr]
    evaluate(retr, "hibrit+rerank")
    out = Path(os.environ.get("RAG_RUNS", "runs")) / "kb" / "rerank_scores_20q.json"; out.write_text(json.dumps(scores_log, ensure_ascii=False, indent=1))
    print("rerank skorları kaydedildi (eşik YOK; Adım 12'de ayrı kümede belirlenir):", out)

if __name__ == "__main__":
    a = sys.argv[1]; kw = {}
    if a == "tokens":
        pos = [x for x in sys.argv[2:] if not x.startswith("--")]
        if pos: kw["tok_id"] = pos[0]
        if "--core" in sys.argv: kw["core"] = True
    globals()[a](**kw)
