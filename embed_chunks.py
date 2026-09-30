#!/usr/bin/env python3
"""
VectorRAG - Part 5: Local Embedding Pipeline (Qwen3-Embedding-4B)
================================================================
Generates dense embeddings for every retrieval child chunk using the official
Qwen/Qwen3-Embedding-4B model (2560-dim, normalized, cosine semantics).

Model acquisition:
  1. Hugging Face (preferred): huggingface_hub.snapshot_download into a local
     cache OUTSIDE the Git repository (default ~/.cache/huggingface).
  2. ModelScope (fallback): the official Qwen mirror on ModelScope.
  The model is downloaded automatically on first run and reused from cache.
  Weights are never committed to Git.

Behaviour in a constrained environment: if the model cannot be acquired or
executed, the pipeline prints concrete diagnostics and STOPS with a non-zero
exit code. It never fabricates embeddings and never silently substitutes a
different model.

Embedding inputs are the chunk `retrieval_text` fields (Title/Section/Document
prefix + content). Document embeddings use NO instruction prefix (per official
Qwen3-Embedding guidance). Query embeddings (later, during retrieval) must use
TASK_INSTRUCTION with the same model.

Caching/idempotency: cache key =
  sha256(model + revision + embedding-config + sha256(retrieval_text))
unchanged chunks are never recomputed; a changed chunk is recomputed alone.

Usage:
  python3 embed_chunks.py --plan-only     # parse inputs, report cache plan
  python3 embed_chunks.py --smoke-test    # 5 chunks from different categories
  python3 embed_chunks.py                 # embed all child chunks
Environment variables:
  EMBEDDING_DEVICE        auto|cpu|cuda|cuda:N|mps   (default: auto)
  EMBEDDING_BATCH_SIZE    int                        (default: conservative auto)
  EMBEDDING_CACHE_DIR     model cache dir            (default: ~/.cache/huggingface)
  EMBEDDING_INPUT_DIR     dir with children/parents.jsonl   (default: data/processed/chunks)
  EMBEDDING_OUTPUT_DIR    output dir                 (default: data/processed/embeddings)
  EMBEDDING_MAX_RETRIES   per-batch retries          (default: 3)
  HF_ENDPOINT             override HF endpoint if mirroring
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODELSCOPE_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
DIMENSION = 2560

TASK_INSTRUCTION = (
    "Given a question about SAP S/4HANA Utilities and Meter-to-Cash "
    "documentation, retrieve the most relevant authoritative passages that "
    "directly answer the question, including relevant business entities, "
    "technical entities, processes, relationships, procedures, and SAP "
    "terminology."
)

EMBEDDING_CONFIG = {
    "dimension": DIMENSION,
    "normalized": True,
    "similarity": "cosine",
    "input_field": "retrieval_text",
    "document_instruction": None,   # official guidance: none for documents
    "query_instruction": TASK_INSTRUCTION,
    "truncation": None,             # 32K context; chunk texts are far below
    "pipeline_version": "1.0",
}

REQUIRED_MEMORY_GB = 9          # bf16 weights (~8GB) + activations overhead
DEFAULT_BATCH_CPU = 4


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(model: str, revision: str, retrieval_text_hash: str) -> str:
    cfg = json.dumps(EMBEDDING_CONFIG, sort_keys=True)
    return hashlib.sha256(
        f"{model}|{revision}|{cfg}|{retrieval_text_hash}".encode()).hexdigest()


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v and v.isdigit() else default


# ------------------------------------------------------------- environment --

def free_ram_gb() -> float:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.strip().split()[0]) / (1024 ** 2)
        return info.get("MemAvailable", 0.0)
    except Exception:
        return -1.0


def resolve_device() -> str:
    dev = os.environ.get("EMBEDDING_DEVICE", "auto")
    if dev != "auto":
        return dev
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def auto_batch_size(device: str) -> int:
    bs = os.environ.get("EMBEDDING_BATCH_SIZE")
    if bs and bs.isdigit():
        return int(bs)
    if device.startswith("cuda"):
        return 16
    ram = free_ram_gb()
    if ram < 0:
        return DEFAULT_BATCH_CPU
    return max(1, min(DEFAULT_BATCH_CPU, int(ram // 2)))


def select_model_source() -> tuple:
    """Return (local_model_dir, provider, revision). Downloads if needed.

    HF first, ModelScope fallback. Raises RuntimeError with concrete
    diagnostics when both are unavailable.
    """
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR",
                               str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HOME", cache_dir)

    revision = "unresolved"

    # --- Hugging Face (preferred) ---
    try:
        from huggingface_hub import HfApi, snapshot_download
        try:
            info = HfApi().model_info(MODEL_NAME)
            revision = info.sha or "unpinned"
            print(f"[acquire] Hugging Face reachable, pinned revision {revision}",
                  flush=True)
        except Exception as e:
            print(f"[acquire] HfApi model_info failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        local = snapshot_download(
            MODEL_NAME, revision=revision if revision != "unresolved" else None,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                            "vocab*", "merges*", "special_tokens_map.json",
                            "configuration*"],
        )
        print(f"[acquire] model available at {local}", flush=True)
        return local, "huggingface", revision
    except Exception as e:
        print(f"[acquire] Hugging Face download failed: "
              f"{type(e).__name__}: {e}", flush=True)

    # --- ModelScope (fallback) ---
    try:
        from modelscope import snapshot_download as ms_snapshot
        local = ms_snapshot(MODELSCOPE_MODEL_ID)
        print(f"[acquire] ModelScope fallback succeeded: {local}", flush=True)
        return local, "modelscope", revision
    except Exception as e:
        print(f"[acquire] ModelScope fallback failed: "
              f"{type(e).__name__}: {e}", flush=True)

    raise RuntimeError(
        "Model acquisition impossible: neither huggingface.co nor "
        "modelscope.cn is reachable and no local cache of "
        f"{MODEL_NAME} exists (cache dir: {cache_dir}).")


def load_model():
    """Acquire + load. Returns (model, provider, revision, device, batch)."""
    device = resolve_device()
    batch = auto_batch_size(device)

    # memory guard BEFORE attempting load (concrete numbers, hard stop)
    if not device.startswith("cuda") and device != "mps":
        ram = free_ram_gb()
        if 0 <= ram < REQUIRED_MEMORY_GB:
            raise RuntimeError(
                f"STOP: insufficient memory to execute {MODEL_NAME}. "
                f"Required ~{REQUIRED_MEMORY_GB}GB (bf16 weights ~8GB + "
                f"activations); available RAM {ram:.1f}GB; GPU: none. "
                f"The pipeline does not substitute a smaller model.")

    local_dir, provider, revision = select_model_source()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(local_dir, device=device)
    return model, provider, revision, device, batch


# ------------------------------------------------------------------ inputs --

def load_inputs():
    in_dir = Path(os.environ.get("EMBEDDING_INPUT_DIR",
                                 str(ROOT / "data/processed/chunks")))
    children = [json.loads(l) for l in
                (in_dir / "children.jsonl").read_text().splitlines() if l.strip()]
    parents = {json.loads(l)["parent_id"]: json.loads(l) for l in
               (in_dir / "parents.jsonl").read_text().splitlines() if l.strip()}
    return children, parents


def build_plan(children):
    """Compute hashes + cache keys for all chunks."""
    plan = []
    for c in children:
        rth = sha256_text(c["retrieval_text"])
        sth = sha256_text(c["source_text"])
        plan.append({
            "chunk": c,
            "retrieval_text_hash": rth,
            "source_text_hash": sth,
            "key_inputs": (MODEL_NAME, revision_str(), rth),
        })
    return plan


_REVISION = {"value": "unresolved"}


def revision_str() -> str:
    return _REVISION["value"]


# ------------------------------------------------------------------ encode --

def encode_texts(model, texts, batch):
    """Embed with single normalization (model-side). Returns list of lists."""
    vecs = model.encode(texts, batch_size=batch,
                        normalize_embeddings=True,
                        show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_with_retry(model, items, batch, max_retries):
    """items: list of plan entries. Returns (embedded list, failures list).
    Batch-level retries; on persistent batch failure, falls back to
    per-chunk encoding so one bad chunk never blocks the rest."""
    embedded, failures = [], []
    def do(batch_items):
        vecs = encode_texts(model, [it["chunk"]["retrieval_text"]
                                    for it in batch_items], batch)
        return vecs
    i = 0
    while i < len(items):
        batch_items = items[i:i + batch]
        t0 = time.time()
        try:
            vecs = do(batch_items)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            ok = False
            for attempt in range(1, max_retries + 1):
                time.sleep(min(2 ** attempt, 8))
                try:
                    vecs = do(batch_items)
                    ok = True
                    break
                except Exception as e2:
                    err = f"{type(e2).__name__}: {e2}"
            if not ok:
                # per-chunk fallback: isolate failures
                print(f"[warn] batch failed after {max_retries} retries "
                      f"({err}); retrying chunk-by-chunk", flush=True)
                vecs = []
                for it in batch_items:
                    try:
                        vecs.append(encode_texts(model,
                                                 [it["chunk"]["retrieval_text"]],
                                                 1)[0])
                    except Exception as e3:
                        failures.append({
                            "chunk_id": it["chunk"]["chunk_id"],
                            "error": f"{type(e3).__name__}: {e3}",
                            "attempts": max_retries + 1})
            else:
                print(f"[warn] batch recovered on retry {attempt}", flush=True)
        if vecs:
            for it, v in zip(batch_items, vecs):
                embedded.append((it, v))
            print(f"[embed] {i + len(batch_items)}/{len(items)} chunks "
                  f"({round(time.time() - t0, 1)}s for {len(batch_items)})",
                  flush=True)
        i += len(batch_items)
    return embedded, failures


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute everything, ignoring the cache")
    args = ap.parse_args()

    out_dir = Path(os.environ.get("EMBEDDING_OUTPUT_DIR",
                                  str(ROOT / "data/processed/embeddings")))
    emb_path = out_dir / "embeddings.jsonl"
    err_path = out_dir / "embedding_errors.json"
    manifest_path = out_dir / "embedding_manifest.json"

    children, parents = load_inputs()
    print(f"[inputs] {len(children)} child chunks from "
          f"{len({c['document_id'] for c in children})} documents", flush=True)

    # ---------- smoke test selection (5 categories) ----------
    smoke_ids = []
    if args.smoke_test:
        seen_cats, chosen = [], []
        for c in children:
            cat = parents.get(c["parent_id"], {}).get("metadata", {}).get("category")
            if cat and cat not in seen_cats:
                seen_cats.append(cat)
                chosen.append((c, cat))
            if len(chosen) == 5:
                break
        smoke_ids = [c["chunk_id"] for c, _ in chosen]
        print(f"[smoke] categories covered: {seen_cats}", flush=True)

    # ---------- existing cache ----------
    cache = {}
    if emb_path.exists() and not args.force:
        for line in emb_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["chunk_id"]] = rec
        print(f"[cache] {len(cache)} existing embeddings loaded", flush=True)
    else:
        print("[cache] no existing embeddings (or --force)", flush=True)

    # ---------- plan-only path: no model needed ----------
    if args.plan_only:
        hits = misses = 0
        for c in children:
            rth = sha256_text(c["retrieval_text"])
            k = cache_key(MODEL_NAME, revision_str(), rth)
            rec = cache.get(c["chunk_id"])
            if rec is not None and rec.get("cache_key") == k:
                hits += 1
            else:
                misses += 1
        print(json.dumps({"plan_only": True, "chunks": len(children),
                          "cache_hits": hits, "would_embed": misses,
                          "note": "plan-only uses 'unresolved' revision; a "
                                  "pinned revision from a live run is part of "
                                  "the real cache key"}, indent=1))
        return 0

    # ---------- model load ----------
    try:
        model, provider, revision, device, batch = load_model()
    except Exception as e:
        print(f"\nSTOP: {e}", flush=True)
        print("No embeddings were generated. No fallback model was used. "
              "Part 5 target remains Qwen/Qwen3-Embedding-4B.", flush=True)
        return 2
    _REVISION["value"] = revision
    print(f"[model] {MODEL_NAME} provider={provider} revision={revision} "
          f"device={device} batch={batch}", flush=True)

    # ---------- select chunks to embed ----------
    targets = []
    hits = 0
    for c in children:
        rth = sha256_text(c["retrieval_text"])
        k = cache_key(MODEL_NAME, revision, rth)
        rec = cache.get(c["chunk_id"])
        if rec is not None and rec.get("cache_key") == k:
            hits += 1
        else:
            targets.append(c)
    print(f"[plan] cache_hits={hits} to_embed={len(targets)}", flush=True)

    if args.smoke_test:
        targets = [c for c in targets if c["chunk_id"] in smoke_ids] or \
                  [c for c in children if c["chunk_id"] in smoke_ids]
        print(f"[smoke] embedding {len(targets)} chunks", flush=True)

    t0 = time.time()
    embedded, failures = [], []
    if targets:
        plan_entries = []
        for c in targets:
            plan_entries.append({
                "chunk": c,
                "retrieval_text_hash": sha256_text(c["retrieval_text"]),
                "source_text_hash": sha256_text(c["source_text"]),
            })
        embedded, failures = embed_with_retry(model, plan_entries, batch,
                                              env_int("EMBEDDING_MAX_RETRIES", 3))
    elapsed = time.time() - t0

    # ---------- smoke test verification ----------
    if args.smoke_test and embedded:
        import math
        print("\n[smoke] verification:", flush=True)
        for it, v in embedded:
            n = math.sqrt(sum(x * x for x in v))
            finite = all(math.isfinite(x) for x in v)
            print(f"  {it['chunk']['chunk_id']}: dim={len(v)} finite={finite} "
                  f"L2={n:.6f} min={min(v):.4f} max={max(v):.4f}", flush=True)
        print("[smoke] pairwise cosine matrix:", flush=True)
        for a in range(len(embedded)):
            row = []
            for b in range(len(embedded)):
                va, vb = embedded[a][1], embedded[b][1]
                cos = sum(x * y for x, y in zip(va, vb))
                row.append(f"{cos: .4f}")
            print("   " + " ".join(row), flush=True)
        print("[smoke] numerical behaviour verified; no retrieval-quality "
              "claim is made from this test.", flush=True)
        return 0

    # ---------- persist ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    if embedded:
        records = [rec for rec in cache.values()]  # keep cache hits
        merged = {r["chunk_id"]: r for r in records}
        for it, v in embedded:
            c = it["chunk"]
            merged[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "parent_id": c["parent_id"],
                "document_id": c["document_id"],
                "section_id": c["section_id"],
                "vector": v,
                "dimension": len(v),
                "normalized": True,
                "model": MODEL_NAME,
                "model_revision": revision,
                "source_text_hash": it["source_text_hash"],
                "retrieval_text_hash": it["retrieval_text_hash"],
                "cache_key": cache_key(MODEL_NAME, revision,
                                       it["retrieval_text_hash"]),
            }
        final = sorted(merged.values(), key=lambda r: r["chunk_id"])
        tmp = emb_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in final))
        os.replace(tmp, emb_path)

    if failures:
        err_path.write_text(json.dumps(failures, indent=1))
        print(f"[fail] {len(failures)} chunk(s) failed; see {err_path}",
              flush=True)

    manifest = {
        "generated_at": utcnow(),
        "model_name": MODEL_NAME,
        "model_revision": revision,
        "acquisition_provider": provider,
        "embedding_dimension": DIMENSION,
        "normalization": "unit L2 (model-side, single pass)",
        "device": device,
        "batch_size": batch,
        "number_of_chunks": len(children),
        "number_embedded": len(embedded),
        "number_failed": len(failures),
        "cache_hits": hits,
        "cache_misses": len(targets),
        "total_time_seconds": round(elapsed, 1),
        "average_time_per_chunk_seconds": round(elapsed / len(embedded), 3)
                                          if embedded else None,
        "embedding_config": EMBEDDING_CONFIG,
        "cache_key_formula": "sha256(model | revision | config | sha256(retrieval_text))",
        "task_instruction": TASK_INSTRUCTION,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)
    print(json.dumps({k: manifest[k] for k in
                      ("number_of_chunks", "number_embedded", "number_failed",
                       "cache_hits", "cache_misses", "total_time_seconds")},
                     indent=1), flush=True)

    if failures:
        print("EMBED_PARTIAL (validation must fail on coverage)", flush=True)
        return 1
    print("EMBED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
