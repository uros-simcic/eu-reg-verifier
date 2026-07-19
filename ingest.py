"""Build the FAISS index from the committed regulation text.

Reads config.json, parses each active regulation into per-article chunks
(parse.py), embeds each article with Mistral's `mistral-embed` over plain HTTPS,
and writes one FAISS index plus a parallel metadata file under index/.

Runs once, offline. This is the only place that spends embedding quota, so it
calls mistral-embed one article at a time, throttles to stay under the
1.00 req/sec embed limit, and logs the rate-limit headers on every response so
the real limits are observed empirically rather than assumed.
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv

from parse import parse_eurlex_html

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"

# Embed limit is 1.00 req/sec; keep a margin so a burst never trips it.
MIN_SECONDS_BETWEEN_CALLS = 1.2
# Substrings that flag a rate-limit header, whatever exact names Mistral returns.
RATELIMIT_HINTS = ("ratelimit", "rate-limit", "retry-after")


def log_rate_limit_headers(resp):
    """Print any rate-limit-related headers so we can see the enforced caps."""
    hits = {k: v for k, v in resp.headers.items()
            if any(h in k.lower() for h in RATELIMIT_HINTS)}
    print("   rate-limit headers:", hits or "none on this response")


# connect fast or fail fast; a slow read is capped so one hung call can never
# pin the app for minutes (that turns into a polite "try again" upstream)
REQUEST_TIMEOUT = (10, 20)


def api_call(request_fn):
    """Run one Mistral request; retry once on 429/5xx or a failed connection.

    Lives next to the HTTP calls so ingestion and query time share the same
    protection -- a transient hiccup shouldn't kill a 99-call ingest run or
    surface as a traceback mid-demo. A read timeout is deliberately NOT
    retried: if the network path is that slow, retrying doubles the user's
    wait for the same outcome. Anything else, or a second failure, raises.
    """
    try:
        return request_fn()
    except requests.ConnectionError:
        time.sleep(2)
        return request_fn()
    except requests.HTTPError as exc:
        resp = exc.response
        if resp is not None and resp.status_code in (429, 500, 502, 503, 504):
            try:
                wait = float(resp.headers.get("Retry-After", 2))
            except ValueError:
                wait = 2.0  # Retry-After may be an HTTP-date; don't crash the retry
            time.sleep(min(wait, 10))
            return request_fn()
        raise


def embed_text(text, api_key, model):
    """Embed one string with mistral-embed; return the vector as list[float]."""
    def call():
        resp = requests.post(
            MISTRAL_EMBED_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": [text]},
            timeout=REQUEST_TIMEOUT,
        )
        log_rate_limit_headers(resp)
        resp.raise_for_status()
        return resp

    return api_call(call).json()["data"][0]["embedding"]


def build():
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY not found in environment/.env")

    with open(ROOT / "config.json") as fh:
        cfg = json.load(fh)
    embed_model = cfg["models"]["embedding"]

    # 1) Parse every active regulation into one flat, ordered list of chunks.
    chunks = []
    for reg_id in cfg["active_regulations"]:
        reg = cfg["regulations"][reg_id]
        reg_chunks = parse_eurlex_html(reg_id, str(ROOT / reg["source_file"]))
        print(f"{reg_id}: parsed {len(reg_chunks)} articles")
        chunks.extend(reg_chunks)

    # 2) Embed each article, sequentially and throttled.
    print(f"\nembedding {len(chunks)} chunks with {embed_model} "
          f"(~{MIN_SECONDS_BETWEEN_CALLS}s apart)...")
    vectors = []
    for i, chunk in enumerate(chunks, 1):
        start = time.monotonic()
        # Embed title + body: the title carries the concept (e.g. "right to erasure").
        text = f"{chunk['title']}\n\n{chunk['text']}"
        print(f"[{i}/{len(chunks)}] {chunk['reg']} art.{chunk['article']}  {chunk['title'][:50]}")
        vectors.append(embed_text(text, api_key, embed_model))
        elapsed = time.monotonic() - start
        if i < len(chunks) and elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    # 3) Build a cosine-similarity index (normalize vectors -> inner product).
    mat = np.array(vectors, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    print(f"\nbuilt index: {index.ntotal} vectors, dim {mat.shape[1]}")

    # 4) Persist index + parallel metadata (metadata order matches index order).
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "regulations.faiss"))
    with open(INDEX_DIR / "metadata.json", "w") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"wrote {INDEX_DIR / 'regulations.faiss'} and {INDEX_DIR / 'metadata.json'}")


if __name__ == "__main__":
    build()
