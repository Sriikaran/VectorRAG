# Embedding Model — Qwen3-Embedding-4B (reproducibility notes)

Target model for VectorRAG Part 5. Weights are **never** committed to Git;
the model is downloaded automatically on first run and reused from a local
cache that lives outside the repository.

## Exact model

- ID: `Qwen/Qwen3-Embedding-4B`
- Output: 2560 dimensions, normalized (unit L2), cosine similarity
- Context: 32K tokens (chunk retrieval texts are far below this; no truncation)
- Document embeddings: **no** instruction prefix (official guidance)
- Query embeddings (later retrieval stage): use `TASK_INSTRUCTION` from
  `embed_chunks.py` with the same model

## How it is downloaded (automatic on first run)

`embed_chunks.py` acquires the model in this order:

1. **Hugging Face (preferred)** — `huggingface_hub.snapshot_download("Qwen/Qwen3-Embedding-4B")`;
   the resolved commit SHA is pinned into every embedding record and into the
   cache key.
2. **ModelScope (fallback)** — official Qwen mirror via the `modelscope`
   package (`snapshot_download("Qwen/Qwen3-Embedding-4B")`).

If neither endpoint is reachable and no local cache exists, the pipeline stops
with concrete diagnostics (exit code 2). It never fabricates embeddings and
never silently substitutes a different model.

## Expected local cache location

- Default: `~/.cache/huggingface` (outside the Git repository; `HF_HOME` set
  by the pipeline if unset)
- Override: `EMBEDDING_CACHE_DIR=/path/to/cache`
- Pre-seeding: download once on a networked machine and copy the HF cache dir
  (or `snapshot_download(..., local_dir=...)`) to the target host.

## Required Python / library versions

- Python ≥ 3.10
- `transformers >= 4.51`
- `sentence-transformers >= 2.7`
- `torch` (CPU or CUDA build)
- `huggingface_hub` (HF route) / `modelscope` (fallback route)

All are available on PyPI. GPU is not required; the device is resolved
automatically and can be forced with `EMBEDDING_DEVICE=cpu|cuda|cuda:N|mps`.

## GPU/CPU behaviour and memory

- `EMBEDDING_DEVICE=auto` (default): CUDA → Apple MPS → CPU.
- Qwen3-Embedding-4B in bf16 needs **~8GB for weights alone**, plus
  activations — plan for **≥9–12GB** of accelerator memory (GPU) or system
  RAM (CPU). The pipeline hard-stops with exact numbers when available memory
  is below `REQUIRED_MEMORY_GB` (9GB) and no GPU/MPS exists; it does not
  substitute a smaller model.
- Batch size: `EMBEDDING_BATCH_SIZE` (default: conservative auto — 16 on GPU,
  scaled from free RAM on CPU; 4GB RAM ⇒ 1–2).

## Fallback acquisition route (HF unavailable)

`pip install modelscope` then re-run `embed_chunks.py` — the pipeline falls
back to ModelScope automatically. If that host is also unreachable (e.g. a
PyPI-only sandbox), the only offline route is copying a cache prepared
elsewhere (see "Pre-seeding" above). Note recorded for the VectorRAG sandbox
as of 2026-09-30: huggingface.co, cdn-lfs.huggingface.co and modelscope.cn
were all unreachable and the host had 3GB RAM / no GPU, so Part 5 produced no
embeddings there; the pipeline code is the deliverable and runs unchanged in a
capable environment.
