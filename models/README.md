# Embedding Models — reproducibility notes

Current build default: **Qwen3-Embedding-0.6B via the ONNX implementation
package `qwen3-embed` (INT8)**. Weights are **never** committed to Git.

## Default profile (all current embedding generation + validation)

- Model (recorded name): `Qwen3-Embedding-0.6B-ONNX`
- Package catalog id: `n24q02m/Qwen3-Embedding-0.6B-ONNX` (the qwen3-embed
  package's Hugging Face repo for this ONNX export)
- Backend: ONNX Runtime (`onnxruntime`), CPU-first; GPU optional via
  `EMBEDDING_DEVICE=AUTO|CUDA` if the backend detects an accelerator
- Quantization: **INT8 dynamic** (`onnx/model_quantized.onnx`, ~0.57GB)
- Dimensions: **1024** (native; MRL 32–1024 supported by the model but not
  used — do not reduce in this build)
- Normalization: package-side unit L2 (verified by the pipeline/validator,
  never applied twice); cosine similarity semantics
- Context: retrieval texts are validated to fit **1024 Qwen tokens** (observed
  max across the corpus: 806) — intentional reduction from the model's 32K
  maximum; no silent truncation
- License: Apache-2.0
- Package: `qwen3-embed` 1.14.0 (`pip install qwen3-embed`, pulls
  `onnxruntime`, `tokenizers`, `numpy`, `huggingface-hub`, `loguru`)

### Automatic acquisition (network available)

`python3 embed_chunks.py` downloads the INT8 ONNX model from the catalog repo
via `huggingface_hub` on first run and caches it
(default `~/.cache/qwen3_embed`-style local cache outside Git; override with
`EMBEDDING_CACHE_DIR`).

Exact offline download command (any networked machine):

    pip install "huggingface_hub" "qwen3-embed==1.14.0"
    hf download n24q02m/Qwen3-Embedding-0.6B-ONNX
    # or: huggingface-cli download n24q02m/Qwen3-Embedding-0.6B-ONNX

### LOCAL MODEL PATH (restricted/offline environments)

Set `EMBEDDING_MODEL_PATH` to a directory laid out exactly like the repo:

    <EMBEDDING_MODEL_PATH>/
        config.json                     (required)
        tokenizer.json                  (required)
        tokenizer_config.json           (required; carries model_max_length)
        special_tokens_map.json         (required special-tokens map)
        onnx/model_quantized.onnx       (required, INT8 weights, ~0.57GB)

Then run:

    EMBEDDING_MODEL_PATH=/absolute/path/to/Qwen3-Embedding-0.6B-ONNX \
    EMBEDDING_DEVICE=CPU \
    EMBEDDING_BATCH_SIZE=2 \
    python3 embed_chunks.py

The pipeline validates the layout, bypasses all network access
(`specific_model_path` + `local_files_only`), loads via ONNX Runtime with
2 CPU threads, and records `model_revision: "local"`.

### CPU/memory behaviour

- INT8 0.6B needs ~0.6GB weights + runtime overhead: comfortably runs in
  ~2.5–3GB RAM (the pipeline hard-stops below `REQUIRED_MIN_FREE_RAM_GB`).
- `EMBEDDING_THREADS` (default 2) controls ONNX Runtime CPU threading.
- `EMBEDDING_BATCH_SIZE` (default 2) is recorded; the 0.6B causal-LM ONNX
  graph pins the effective batch to 1 (manifest records both).

## Recorded environment blocker (Arena sandbox, 2026-09-30)

`huggingface.co` (and cdn-lfs/modelscope) are unreachable from the sandbox
(TLS connection closed; only PyPI is reachable), so automatic acquisition
fails with `ConnectError: TLS/SSL connection has been closed (EOF)`. No local
model has been supplied yet, therefore **0/453 embeddings exist in this
environment** and `validate_embeddings.py` honestly reports BLOCKED. Supply
the model locally (see above) or run on a networked machine to produce them.

## Optional high-memory quality profile (NOT the default)

- Model: `Qwen/Qwen3-Embedding-4B` — 2560-dim, bf16, ~9–12GB memory, GPU
  recommended; run via sentence-transformers (`transformers >= 4.51`,
  `sentence-transformers >= 2.7`). Kept documented from the previous Part 5
  implementation (git history, `embed_chunks.py` v1); NOT used by the current
  default pipeline. Embedding records carry `model`/`backend` so artifacts
  from either profile are distinguishable.
