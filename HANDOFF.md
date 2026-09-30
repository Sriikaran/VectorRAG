# HANDOFF — SAP Utilities Evidence-Graph RAG

Handoff checkpoint for development continuing in Antigravity.
Everything below reflects the repository state at this commit (verified by
running all validation scripts immediately before committing).

## CURRENT STATUS

| Part | Scope | Status |
|------|-------|--------|
| 1 | Corpus inventory + PDF ingestion | ✅ complete, validated (7/7) |
| 2 | OCR / text recovery | ✅ complete, validated (11/11) |
| 3 | Cleaning + canonical knowledge representation | ✅ complete, validated (12/12) |
| 4 | Structure-aware parent/child chunking | ✅ complete, validated (20/20) |
| 5 | Embedding pipeline | ⚠️ **pipeline complete; 0/453 embeddings generated** (Hugging Face unreachable from the Arena sandbox; validator honestly reports BLOCKED) |

No embeddings exist. No embeddings were fabricated. No substitute embedding
model was used. Part 5 is NOT fully complete until 453 real vectors exist and
`validate_embeddings.py` passes.

## ARCHITECTURE (target)

```
29 SAP PDFs
    ↓
PDF ingestion                 ingest.py            ✅ Part 1
    ↓
OCR                           ocr_pipeline.py      ✅ Part 2
    ↓
canonical representation      canonicalize.py      ✅ Part 3
    ↓
structure-aware chunks        chunker.py           ✅ Part 4
    ↓
Qwen3-Embedding-0.6B-ONNX     embed_chunks.py      ⚠️ Part 5 (code ready, 0/453 vectors)
    ↓
Qdrant                                             ← NEXT (Part 6)
    ↓
hybrid retrieval
    ↓
evidence graph
    ↓
reranker
    ↓
external LLM API
    ↓
citation validation
    ↓
API/UI/evaluation
```

## EXACT NEXT STEP — Part 5.1

Provide the Qwen3-Embedding-0.6B-ONNX model locally and generate/validate the
453 embeddings:

1. On any networked machine:
   `huggingface-cli download n24q02m/Qwen3-Embedding-0.6B-ONNX`
   (or `hf download ...`); copy the repo snapshot to the dev machine.
2. Required local layout (validated automatically by the pipeline):
   ```
   <EMBEDDING_MODEL_PATH>/
       config.json
       tokenizer.json
       tokenizer_config.json
       special_tokens_map.json
       onnx/model_quantized.onnx    (INT8, ~0.57GB)
   ```
3. Run:
   ```bash
   EMBEDDING_MODEL_PATH=/absolute/path/to/Qwen3-Embedding-0.6B-ONNX \
   EMBEDDING_DEVICE=CPU EMBEDDING_BATCH_SIZE=2 python3 embed_chunks.py
   ```
   (first `--smoke-test`, then the full run; both are cache-driven and
   idempotent)
4. Validate: `python3 validate_embeddings.py` — must report 453/453, dim 1024,
   unit-normalized, hashes matching current chunks.

## THEN — Part 6

Qdrant vector + sparse index (build the collection from
`data/processed/embeddings/embeddings.jsonl`; dense 1024-dim cosine + sparse
components). Do not start Qdrant before Part 5.1 is green.

## WHAT IS DONE (verified at this commit)

- **Part 1**: 29 documents ingested via PyMuPDF with curated metadata
  (`corpus_inventory.json` → every record). `data/processed/documents/`,
  `manifest.json`, `parse_errors.json` (empty).
- **Part 2**: all 501 pages OCR'd (tesserocr 2.11 / Tesseract 5.5.1 LSTM,
  full-page 300dpi renders, mean confidence 92.5, 998k chars). Word boxes +
  confidence preserved. `data/processed/ocr/`, `ocr_manifest.json`. Language
  data vendored in `tessdata/` (Apache-2.0) so OCR reproduces offline.
- **Part 3**: canonical layer — per-page `raw_text` (byte-identical OCR
  evidence) + `cleaned_text`, noise spans marked (never deleted), blocks
  (headings/paragraphs/lists/tables/notes/examples/fragments), 1,834-section
  hierarchy with stable IDs and paths, authority model 3/2/1.
  `data/processed/canonical/`, `documents.jsonl`, `canonical_manifest.json`.
- **Part 4**: 1,834 parents + 453 children; structure-first packing
  (section subtrees never crossed; small sibling chunks merged under common
  parent; oversized units split at sentence/item/row boundaries; table chunks
  always repeat headers); real Qwen3 BPE token counts (avg 425 / median 457 /
  max 774); deterministic IDs (`doc_XXX_sec_NNN_chunk_KKK`);
  `source_text` vs `retrieval_text` (metadata-only context prefix) separated.
  `data/processed/chunks/`.

## WHAT IS NOT DONE

- **Embeddings** (Part 5.1 above) — nothing in `data/processed/embeddings/`.
- Everything from Qdrant onward (Part 6+): vector DB, sparse retrieval,
  graph, reranker, LLM, API/UI, citation validation, evaluation.

## KEY MODEL / DESIGN DECISIONS

- **Embedding model (current build)**: Qwen3-Embedding-**0.6B**-ONNX, INT8,
  **1024-dim**, normalized, cosine, CPU-first, configured batch 2 (effective
  ONNX batch 1 for this causal-LM graph). Package: `qwen3-embed` 1.14.0.
  Local-model mechanism: `EMBEDDING_MODEL_PATH` (bypasses network via
  `specific_model_path` + `local_files_only`).
- **Optional high-memory profile** (documented in `models/README.md`, not
  default): Qwen3-Embedding-4B, 2560-dim, sentence-transformers, ~9–12GB
  RAM/GPU.
- **Queries** must embed with `TASK_INSTRUCTION` (constant in
  `embed_chunks.py`) via `query_embed(task=...)` — same model, same dimension.
  Document embeddings use NO instruction prefix.
- **Context**: chunk `retrieval_text`s validated ≤1024 Qwen tokens
  (max observed 806). Chunk hard max 800 tokens by construction.
- **Idempotency everywhere**: content hashes (PDF sha256 → page hashes →
  source_page_hash → retrieval_text hash) make every pipeline rerun a no-op
  unless inputs changed. Embedding cache key:
  `sha256(model | backend | revision | config | sha256(retrieval_text))`.
- **Evidence preservation**: raw OCR text is never modified after Part 2
  (canonical check 6 enforces byte-identity); uncertain OCR substitutions are
  not corrected; fragments are flagged, kept in parent provenance, never
  become retrieval chunks; OCR table values are not "corrected".

## IMPORTANT DIRECTORIES

- `data/processed/{documents,ocr,canonical,chunks}` — completed artifacts
  (tracked in Git; ~34MB; largest file <5MB).
- `data/processed/embeddings/` — will be created by Part 5.1.
- `tessdata/` — vendored OCR language data (required by `ocr_pipeline.py`).
- `models/` — documentation only; weights are git-ignored, never committed.

## VALIDATION COMMANDS

```bash
python3 validate_ingestion.py    # Part 1 (7 checks)
python3 validate_ocr.py          # Part 2 (11 checks; reruns OCR pipeline skip-path)
python3 validate_canonical.py    # Part 3 (12 checks)
python3 validate_chunks.py       # Part 4 (20 checks; reruns chunker)
python3 embed_chunks.py --plan-only   # Part 5 plan/config check (no model needed)
python3 validate_embeddings.py   # Part 5 full validation (needs embeddings)
```

Dependencies: `pip install -r requirements.txt` (Python ≥ 3.11; optional
profiles in `requirements-optional.txt`). PDF placement: see `DATA_SETUP.md`
(both PDF sets are tracked in Git, so a normal clone restores them).

## HOW TO CONTINUE (Antigravity)

1. Clone this repository (branch `arena/01a0f133-vectorrag` — complete
   history plus all Part 1–4 artifacts).
2. `pip install -r requirements.txt`.
3. Run the validation commands above to confirm the local state.
4. Execute Part 5.1 (model + embeddings + validation).
5. Proceed to Part 6 (Qdrant vector + sparse index), then the rest of the
   architecture diagram. Parts 1–4 outputs are final; every stage is
   hash-idempotent, so reruns are safe but unnecessary.
