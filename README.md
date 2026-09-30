# VectorRAG — SAP Utilities Evidence-Graph RAG (Data ingestion → Embeddings)

Local RAG foundation over 29 SAP S/4HANA Utilities (Meter-to-Cash) Help Portal
PDFs. Built in five completed stages: ingestion → OCR → canonicalization →
structure-aware chunking → embedding pipeline. **Status and next steps:
see [HANDOFF.md](HANDOFF.md).**

## Repository layout

```
├── 01_..29_*.pdf          29 reference PDFs (curated metadata source; tracked)
├── corpus_inventory.json  master mapping: reference ↔ data PDF, title, category,
│                          m2c_relevance, sha256, size, source URL
├── data/
│   ├── 1.pdf .. 29.pdf    29 scanned data PDFs (tracked; ~196MB total)
│   └── processed/         ALL GENERATED ARTIFACTS (tracked; ~34MB)
│       ├── documents/     Part 1: doc_001..029.json structure extraction
│       ├── manifest.json / parse_errors.json
│       ├── ocr/           Part 2: doc_XXX/page_XXX.json (501 pages, word boxes,
│       │                  confidence) + ocr_manifest.json
│       ├── canonical/     Part 3: doc_XXX/page_XXX.json (raw+cleaned text, blocks,
│       │                  sections) + documents.jsonl + canonical_manifest.json
│       └── chunks/        Part 4: parents.jsonl (1,834) + children.jsonl (453)
│                          + chunk_manifest.json + sample_chunks.json
├── ingest.py              Part 1: ingestion (idempotent, hash-based)
├── validate_ingestion.py  Part 1 validation (7 checks)
├── ocr_pipeline.py        Part 2: OCR (tesserocr/Tesseract 5.5.1 LSTM, 300dpi,
│                          page-hash idempotent)
├── validate_ocr.py        Part 2 validation (11 checks)
├── canonicalize.py        Part 3: cleaning + canonical representation
├── validate_canonical.py  Part 3 validation (12 checks)
├── chunker.py             Part 4: parent/child chunking (Qwen3 BPE token counts)
├── validate_chunks.py     Part 4 validation (20 checks)
├── embed_chunks.py        Part 5: embedding pipeline (Qwen3-Embedding-0.6B-ONNX)
├── validate_embeddings.py Part 5 validation (13 checks; BLOCKED until model
│                          is supplied — see HANDOFF.md Part 5.1)
├── tessdata/              vendored Tesseract language data (offline OCR)
├── models/                model documentation (weights NEVER committed)
├── requirements.txt       core dependencies
└── requirements-optional.txt  optional profiles (4B embeddings, ModelScope)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python ≥ 3.11
pip install -r requirements.txt
```

`tesserocr` ships a bundled Tesseract; the language data is vendored in
`tessdata/`, so OCR runs fully offline. No GPU is required anywhere in the
current pipeline.

## Regeneration / validation commands

Every stage is idempotent (content-hash based) — reruns rewrite nothing
unless inputs changed:

```bash
python3 ingest.py               && python3 validate_ingestion.py   # Part 1
python3 ocr_pipeline.py         && python3 validate_ocr.py         # Part 2
python3 canonicalize.py         && python3 validate_canonical.py   # Part 3
python3 chunker.py              && python3 validate_chunks.py      # Part 4
python3 embed_chunks.py --plan-only && python3 validate_embeddings.py  # Part 5 (plan)
```

Embedding generation requires the Qwen3-Embedding-0.6B-ONNX model — either
network access to Hugging Face (automatic download) or a local copy:

```bash
EMBEDDING_MODEL_PATH=/absolute/path/to/Qwen3-Embedding-0.6B-ONNX \
EMBEDDING_DEVICE=CPU EMBEDDING_BATCH_SIZE=2 python3 embed_chunks.py
```

See `models/README.md` for the model layout, download command, and license.
