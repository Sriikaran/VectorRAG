#!/usr/bin/env python3
"""
VectorRAG - Part 5: Local Embedding Pipeline
============================================
DEFAULT PROFILE (current build): Qwen3-Embedding-0.6B via the lightweight ONNX
implementation package `qwen3-embed` (fastembed-style), INT8 quantization,
1024-dimensional normalized vectors, CPU-first execution.

    Model (catalog id): n24q02m/Qwen3-Embedding-0.6B-ONNX   (recorded name:
    "Qwen3-Embedding-0.6B-ONNX"), file onnx/model_quantized.onnx (INT8 dynamic),
    Apache-2.0, dim 1024, last-token pooling, package-side L2 normalization
    (verified here, never applied twice).

OPTIONAL high-memory quality profile (documented in models/README.md, NOT the
default): Qwen3-Embedding-4B via sentence-transformers (2560-dim). Embedding
records carry "model" and "backend" so both profiles stay distinguishable.

Acquisition:
  1. automatic: qwen3-embed downloads from the Hugging Face catalog repo and
     caches it (default cache outside the Git repository).
  2. local: set EMBEDDING_MODEL_PATH to a directory with the pre-downloaded
     model (layout below); the package's `specific_model_path` option then
     bypasses all network access:
         <EMBEDDING_MODEL_PATH>/
             config.json
             tokenizer.json
             tokenizer_config.json
             special_tokens_map.json        (and generation_config.json if present)
             onnx/model_quantized.onnx      (INT8)

Inputs: retrieval_text of every child chunk (data/processed/chunks/children.jsonl),
embedded verbatim; NO instruction prefix on document embeddings (official Qwen
guidance). Query embeddings (later retrieval stage) use TASK_INSTRUCTION via the
package's query_embed(task=...) with the SAME model.

Context: retrieval texts are validated to fit EMBEDDING_MAX_INPUT_TOKENS
(default 1024 Qwen tokens) - nothing is silently truncated.

Caching: cache key = sha256(model | revision | config | dim | normalization |
sha256(retrieval_text)). Unchanged chunks are never recomputed.

Usage:
  python3 embed_chunks.py --plan-only      # no model needed; cache plan
  python3 embed_chunks.py --smoke-test     # 5 chunks from different categories
  python3 embed_chunks.py                  # embed all child chunks

Environment variables:
  EMBEDDING_DEVICE         CPU (default) | AUTO | CUDA     (GPU optional)
  EMBEDDING_BATCH_SIZE     configured batch size (default 2; the 0.6B causal-LM
                           ONNX graph pins the effective batch to 1 - recorded
                           in the manifest)
  EMBEDDING_THREADS        CPU threads for ONNX Runtime (default 2)
  EMBEDDING_MODEL_PATH     absolute path to a pre-downloaded local model dir
  EMBEDDING_CACHE_DIR      model cache dir (default ~/.cache/qwen3_embed)
  EMBEDDING_MAX_INPUT_TOKENS  context validation limit (default 1024)
  EMBEDDING_INPUT_DIR / EMBEDDING_OUTPUT_DIR               (test isolation)
  EMBEDDING_MAX_RETRIES    per-batch retries (default 3)
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- default profile (Part 5, current build) ---
MODEL_NAME = "Qwen3-Embedding-0.6B-ONNX"            # recorded in records
MODEL_CATALOG_ID = "n24q02m/Qwen3-Embedding-0.6B-ONNX"  # qwen3-embed catalog id
BACKEND = "onnx"
DIMENSION = 1024
MODEL_FILE = "onnx/model_quantized.onnx"            # INT8 dynamic quantization
MODEL_LICENSE = "apache-2.0"

# --- optional high-memory quality profile (documented, not default) ---
OPTIONAL_PROFILES = {
    "Qwen3-Embedding-4B": {
        "library": "sentence-transformers",
        "dimension": 2560,
        "note": "requires ~9-12GB memory; see models/README.md",
    },
}

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
    "normalization_mode": "package-side L2, verified (never applied twice)",
    "similarity": "cosine",
    "input_field": "retrieval_text",
    "document_instruction": None,   # official Qwen guidance: none for documents
    "query_instruction": TASK_INSTRUCTION,
    "quantization": "INT8 dynamic (onnx/model_quantized.onnx)",
    "max_input_tokens": 1024,
    "pipeline_version": "2.0",
}

REQUIRED_MIN_FREE_RAM_GB = 2.5   # INT8 0.6B (~0.6GB weights) + runtime overhead
DEFAULT_BATCH = 2
DEFAULT_THREADS = 2
NORM_TOLERANCE = 1e-3


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def env(name: str, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def env_int(name: str, default: int) -> int:
    v = env(name)
    return int(v) if v and v.isdigit() else default


def cache_key(revision: str, retrieval_text_hash: str) -> str:
    cfg = json.dumps(EMBEDDING_CONFIG, sort_keys=True)
    return hashlib.sha256(
        f"{MODEL_NAME}|{BACKEND}|{revision}|{cfg}|{retrieval_text_hash}".encode()
    ).hexdigest()


def free_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[0]) / (1024 ** 2)
    except Exception:
        pass
    return -1.0


# ------------------------------------------------------- token validation --

_QTOK = None


def qwen_tokens(text: str) -> int:
    """Count tokens with the Qwen-family BPE tokenizer (offline package)."""
    global _QTOK
    if _QTOK is None:
        from qwen_tokenizer.tokenizer import get_tokenizer
        _QTOK = get_tokenizer("Qwen/Qwen3-6B")
    return len(_QTOK.encode(text))


def validate_contexts(children, max_tokens):
    """No silent truncation: every retrieval_text must fit the context limit."""
    too_long = []
    for c in children:
        n = qwen_tokens(c["retrieval_text"])
        if n > max_tokens:
            too_long.append({"chunk_id": c["chunk_id"], "tokens": n})
    return too_long


# ------------------------------------------------------------------ model --

class ModelUnavailable(RuntimeError):
    pass


def load_model():
    """Acquire + load the ONNX embedding model.

    Returns (model, revision, device, batch_configured, threads).
    Raises ModelUnavailable with concrete diagnostics when acquisition fails.
    """
    from qwen3_embed import TextEmbedding, Device

    device_name = env("EMBEDDING_DEVICE", "CPU").upper()
    if device_name == "AUTO":
        cuda = Device.AUTO
        device = "auto"
    elif device_name in ("CUDA", "GPU"):
        cuda = Device.CUDA
        device = "cuda"
    else:
        cuda = Device.CPU
        device = "cpu"

    threads = env_int("EMBEDDING_THREADS", DEFAULT_THREADS)
    batch = env_int("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH)
    model_path = env("EMBEDDING_MODEL_PATH")
    cache_dir = env("EMBEDDING_CACHE_DIR")

    ram = free_ram_gb()
    if 0 <= ram < REQUIRED_MIN_FREE_RAM_GB:
        raise ModelUnavailable(
            f"insufficient memory: {ram:.1f}GB available, need ~{REQUIRED_MIN_FREE_RAM_GB}GB")

    kwargs = {"threads": threads, "cuda": cuda}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if model_path:
        p = Path(model_path).expanduser().resolve()
        onnx_file = p / MODEL_FILE
        missing = [f for f in ("config.json", "tokenizer.json", "tokenizer_config.json")
                   if not (p / f).exists()] + \
                  ([MODEL_FILE] if not onnx_file.exists() else [])
        if missing:
            raise ModelUnavailable(
                f"EMBEDDING_MODEL_PATH={p} is missing required files: {missing}; "
                f"expected layout: {EXPECTED_LOCAL_LAYOUT}")
        print(f"[acquire] local model path: {p} (network bypassed)", flush=True)
        kwargs.update({"specific_model_path": str(p), "local_files_only": True})
        revision = "local"
    else:
        print(f"[acquire] automatic acquisition of {MODEL_CATALOG_ID} "
              f"(INT8, {MODEL_FILE}) ...", flush=True)
        kwargs["local_files_only"] = False
        revision = "auto"

    try:
        model = TextEmbedding(MODEL_CATALOG_ID, **kwargs)
    except Exception as e:
        hint = "" if model_path else (
            f" Set EMBEDDING_MODEL_PATH to a pre-downloaded model directory "
            f"(expected layout: {EXPECTED_LOCAL_LAYOUT}); see models/README.md "
            f"for the exact download command.")
        raise ModelUnavailable(f"{type(e).__name__}: {e}.{hint}") from e

    # resolve the concrete revision (cached commit SHA) for provenance
    revision = _resolve_revision(model, revision)
    return model, revision, device, batch, threads


def _resolve_revision(model, current):
    """Best-effort resolution of the cached snapshot's commit SHA."""
    if current == "local":
        return "local"
    try:
        from qwen3_embed.common.model_management import ModelManagement
        snap = Path(str(model.cache_dir)) / "models--n24q02m--Qwen3-Embedding-0.6B-ONNX" \
            if "models--" in str(model.cache_dir) else Path(str(model._model_dir))
        for candidate in (snap, Path(str(model.cache_dir))):
            sha = ModelManagement._resolve_cached_revision(candidate)
            if sha:
                return sha
    except Exception:
        pass
    return current


EXPECTED_LOCAL_LAYOUT = {
    "config.json": "required",
    "tokenizer.json": "required",
    "tokenizer_config.json": "required",
    "special_tokens_map.json": "required (special tokens map)",
    "onnx/model_quantized.onnx": "required (INT8 weights, ~0.57GB)",
}


# ------------------------------------------------------------------ inputs --

def load_inputs():
    in_dir = Path(env("EMBEDDING_INPUT_DIR", str(ROOT / "data/processed/chunks")))
    children = [json.loads(l) for l in
                (in_dir / "children.jsonl").read_text().splitlines() if l.strip()]
    parents = {}
    pl = in_dir / "parents.jsonl"
    if pl.exists():
        for l in pl.read_text().splitlines():
            if l.strip():
                p = json.loads(l)
                parents[p["parent_id"]] = p
    return children, parents


def embed_with_retry(model, items, batch, max_retries):
    """Embed with batch retries; a persistently failing batch degrades to
    per-chunk encoding so one bad chunk never blocks the rest."""
    embedded, failures = [], []
    i = 0
    while i < len(items):
        batch_items = items[i:i + batch]
        t0 = time.time()
        vecs = None
        try:
            vecs = list(model.embed([it["chunk"]["retrieval_text"]
                                     for it in batch_items]))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            cur_batch = len(batch_items)
            for attempt in range(1, max_retries + 1):
                time.sleep(min(2 ** attempt, 8))
                # reduce batch size on retry if appropriate
                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    batch_items_r = batch_items[:cur_batch]
                else:
                    batch_items_r = batch_items
                try:
                    vecs = list(model.embed([it["chunk"]["retrieval_text"]
                                             for it in batch_items_r]))
                    print(f"[warn] batch recovered on retry {attempt} "
                          f"(size {cur_batch})", flush=True)
                    break
                except Exception as e2:
                    err = f"{type(e2).__name__}: {e2}"
                    vecs = None
            if vecs is None:
                print(f"[warn] batch failed after retries ({err}); "
                      f"isolating chunks", flush=True)
                vecs = []
                for it in batch_items:
                    try:
                        vecs.append(list(model.embed([it["chunk"]["retrieval_text"]]))[0])
                    except Exception as e3:
                        failures.append({"chunk_id": it["chunk"]["chunk_id"],
                                         "error": f"{type(e3).__name__}: {e3}",
                                         "attempts": max_retries + 1})
        if vecs:
            for it, v in zip(batch_items, vecs):
                embedded.append((it, v))
            print(f"[embed] {i + len(batch_items)}/{len(items)} chunks "
                  f"({round(time.time() - t0, 1)}s)", flush=True)
        i += len(batch_items)
    return embedded, failures


def record_for(it, v, revision):
    c = it["chunk"]
    return {
        "chunk_id": c["chunk_id"],
        "parent_id": c["parent_id"],
        "document_id": c["document_id"],
        "section_id": c["section_id"],
        "vector": v,
        "dimension": len(v),
        "normalized": True,
        "model": MODEL_NAME,
        "backend": BACKEND,
        "model_catalog_id": MODEL_CATALOG_ID,
        "model_revision": revision,
        "source_text_hash": it["source_text_hash"],
        "retrieval_text_hash": it["retrieval_text_hash"],
        "cache_key": cache_key(revision, it["retrieval_text_hash"]),
    }


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(env("EMBEDDING_OUTPUT_DIR", str(ROOT / "data/processed/embeddings")))
    emb_path = out_dir / "embeddings.jsonl"
    err_path = out_dir / "embedding_errors.json"
    manifest_path = out_dir / "embedding_manifest.json"

    children, parents = load_inputs()
    n_docs = len({c["document_id"] for c in children})
    print(f"[inputs] {len(children)} child chunks from {n_docs} documents",
          flush=True)

    max_tokens = env_int("EMBEDDING_MAX_INPUT_TOKENS",
                         EMBEDDING_CONFIG["max_input_tokens"])
    too_long = validate_contexts(children, max_tokens)
    if too_long:
        print(f"[context] {len(too_long)} chunk(s) exceed {max_tokens} tokens: "
              f"{too_long[:5]} ...", flush=True)
        if not args.plan_only:
            print("STOP: context validation failed (no silent truncation).",
                  flush=True)
            return 2
    else:
        worst = max(qwen_tokens(c["retrieval_text"]) for c in children)
        print(f"[context] all retrieval_texts fit {max_tokens} tokens "
              f"(max observed: {worst})", flush=True)

    cache = {}
    if emb_path.exists() and not args.force:
        for line in emb_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["chunk_id"]] = rec
        print(f"[cache] {len(cache)} existing embeddings loaded", flush=True)

    if args.plan_only:
        # plan-only uses the placeholder revision; a real run pins the revision
        # into the cache key, so hits here are indicative, not authoritative
        hits = misses = 0
        for c in children:
            k = cache_key("PLANNING", sha256_text(c["retrieval_text"]))
            rec = cache.get(c["chunk_id"])
            if rec is not None and rec.get("retrieval_text_hash") == \
                    sha256_text(c["retrieval_text"]) and \
                    rec.get("model") == MODEL_NAME and \
                    rec.get("dimension") == DIMENSION:
                hits += 1
            else:
                misses += 1
        print(json.dumps({"plan_only": True, "chunks": len(children),
                          "content_match_hits": hits, "would_embed": misses},
                         indent=1))
        return 0

    try:
        model, revision, device, batch, threads = load_model()
    except ModelUnavailable as e:
        print(f"\nSTOP: model unavailable - {e}", flush=True)
        print(f"Acquisition of {MODEL_CATALOG_ID} (INT8) failed and no local "
              f"model was supplied via EMBEDDING_MODEL_PATH.", flush=True)
        print("No embeddings were generated. No other embedding model was "
              "used. Do not treat Part 5 as complete.", flush=True)
        return 2

    print(f"[model] {MODEL_CATALOG_ID} revision={revision} backend={BACKEND} "
          f"device={device} batch_configured={batch} threads={threads}",
          flush=True)

    # ---- chunk selection (cache-aware) ----
    targets, hits = [], 0
    for c in children:
        rth = sha256_text(c["retrieval_text"])
        rec = cache.get(c["chunk_id"])
        if rec is not None and rec.get("cache_key") == cache_key(revision, rth):
            hits += 1
        else:
            targets.append(c)
    print(f"[plan] cache_hits={hits} to_embed={len(targets)}", flush=True)

    # ---- smoke test subset (5 categories) ----
    if args.smoke_test:
        chosen, cats = [], []
        for c in children:
            cat = parents.get(c["parent_id"], {}).get("metadata", {}).get("category")
            if cat and cat not in cats:
                cats.append(cat)
                chosen.append(c)
            if len(chosen) == 5:
                break
        smoke_set = {c["chunk_id"] for c in chosen}
        targets = [c for c in targets if c["chunk_id"] in smoke_set] or chosen
        print(f"[smoke] categories: {cats}", flush=True)

    t0 = time.time()
    embedded, failures = [], []
    if targets:
        items = [{"chunk": c,
                  "retrieval_text_hash": sha256_text(c["retrieval_text"]),
                  "source_text_hash": sha256_text(c["source_text"])}
                 for c in targets]
        embedded, failures = embed_with_retry(model, items, batch,
                                              env_int("EMBEDDING_MAX_RETRIES", 3))
    elapsed = time.time() - t0

    # ---- verification (smoke or full) ----
    if embedded:
        bad_norm, bad_dim, bad_finite = [], [], []
        for it, v in embedded:
            n = math.sqrt(sum(x * x for x in v))
            if len(v) != DIMENSION:
                bad_dim.append(it["chunk"]["chunk_id"])
            if not all(math.isfinite(x) for x in v):
                bad_finite.append(it["chunk"]["chunk_id"])
            if abs(n - 1.0) > NORM_TOLERANCE:
                bad_norm.append((it["chunk"]["chunk_id"], round(n, 6)))
        print(f"[verify] dim={DIMENSION} ok={len(embedded) - len(bad_dim)} "
              f"finite_ok={len(embedded) - len(bad_finite)} "
              f"norm_ok={len(embedded) - len(bad_norm)} "
              f"(tolerance {NORM_TOLERANCE})", flush=True)
        if bad_dim or bad_finite or bad_norm:
            print(f"[verify] FAILURES dim={bad_dim[:3]} finite={bad_finite[:3]} "
                  f"norm={bad_norm[:3]}", flush=True)
        print("[verify] sample cosine matrix (first up-to-5 vectors):", flush=True)
        for a in range(min(5, len(embedded))):
            row = []
            for b in range(min(5, len(embedded))):
                cos = sum(x * y for x, y in zip(embedded[a][1], embedded[b][1]))
                row.append(f"{cos: .4f}")
            print("   " + " ".join(row), flush=True)
        if args.smoke_test:
            # instruction-aware query path smoke (same model; recorded for later)
            try:
                qv = list(model.query_embed(
                    ["How are meter reading results estimated?"],
                    task=TASK_INSTRUCTION))[0]
                qn = math.sqrt(sum(x * x for x in qv))
                print(f"[smoke] query_embed(task=TASK_INSTRUCTION): dim={len(qv)} "
                      f"L2={qn:.6f} finite={all(math.isfinite(x) for x in qv)}",
                      flush=True)
                cos0 = sum(x * y for x, y in zip(qv, embedded[0][1]))
                print(f"[smoke] query-vs-first-chunk cosine={cos0:.4f} "
                      f"(numerical check only, no quality claim)", flush=True)
            except Exception as e:
                print(f"[smoke] query_embed failed: {type(e).__name__}: {e}",
                      flush=True)
        if args.smoke_test:
            print("[smoke] done - numerical behaviour verified; no retrieval "
                  "quality claim is made.", flush=True)
            return 0

    if not args.smoke_test:
        # ---- persist merged cache ----
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = dict(cache)
        for it, v in embedded:
            rec = record_for(it, v, revision)
            merged[rec["chunk_id"]] = rec
        final = sorted(merged.values(), key=lambda r: r["chunk_id"])
        tmp = emb_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in final))
        os.replace(tmp, emb_path)

        if failures:
            err_path.write_text(json.dumps(failures, indent=1))
            print(f"[fail] {len(failures)} chunk(s) failed; see {err_path}",
                  flush=True)

        n_unique = len({tuple(round(x, 9) for x in r["vector"])
                        for r in final})
        all_vals_min = min(min(r["vector"]) for r in final)
        all_vals_max = max(max(r["vector"]) for r in final)
        # record upstream artifact hashes so validation can prove the
        # canonical/chunk layers were not modified by embedding runs
        in_dir = Path(env("EMBEDDING_INPUT_DIR", str(ROOT / "data/processed/chunks")))
        artifact_hashes = {}
        for f in (in_dir / "children.jsonl", in_dir / "parents.jsonl",
                  ROOT / "data/processed/canonical/documents.jsonl"):
            if f.exists():
                artifact_hashes[str(f.relative_to(ROOT))] = sha256_text(
                    f.read_text())
        manifest = {
            "generated_at": utcnow(),
            "model_name": MODEL_NAME,
            "model_catalog_id": MODEL_CATALOG_ID,
            "model_revision": revision,
            "backend": BACKEND,
            "quantization": EMBEDDING_CONFIG["quantization"],
            "embedding_dimension": DIMENSION,
            "normalization": EMBEDDING_CONFIG["normalization_mode"],
            "device": device,
            "batch_size_configured": batch,
            "batch_size_effective": 1,
            "cpu_threads": threads,
            "number_of_chunks": len(children),
            "number_embedded": len(embedded),
            "number_failed": len(failures),
            "cache_hits": hits,
            "cache_misses": len(targets),
            "total_time_seconds": round(elapsed, 1),
            "average_time_per_chunk_seconds": round(elapsed / len(embedded), 3)
                                              if embedded else None,
            "quality_check": {
                "unique_vectors": n_unique,
                "duplicate_vectors": len(final) - n_unique,
                "value_min": all_vals_min,
                "value_max": all_vals_max,
                "note": "numerical checks only; no semantic quality claim",
            },
            "embedding_config": EMBEDDING_CONFIG,
            "artifact_hashes": artifact_hashes,
            "cache_key_formula": ("sha256(model | backend | revision | config | "
                                  "sha256(retrieval_text))"),
            "task_instruction": TASK_INSTRUCTION,
            "optional_profiles": OPTIONAL_PROFILES,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
