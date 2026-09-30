# DATA SETUP — original PDFs

Both PDF sets are **already tracked in this Git repository**, so a normal
clone restores them. This document records the exact expected layout and how
to verify integrity after cloning.

## Expected folder structure

```
<VectorRAG>/
├── 01_Utilities_Master_Data.pdf                              # reference PDF
├── 02_Move_In_Out_Overview.pdf                               # ... (29 files)
├── ...
├── 29_Disconnection_Reconnection_of_a_Utility_Installation.pdf
└── data/
    ├── 1.pdf                                                 # data PDF (scan)
    ├── 2.pdf
    ├── ...
    └── 29.pdf                                                # (29 files)
```

- **Reference PDFs** (project root, `NN_Title.pdf`): SAP Help Portal exports
  used as the curated metadata source (title, category, M2C relevance).
- **Data PDFs** (`data/NN.pdf`): the 29 scanned/image-only SAP documents that
  flow through the pipeline. 1:1 numeric mapping to the reference PDFs
  (`corpus_inventory.json` → `mapping`).

Both sets are tracked in Git in full (data PDFs total ~196MB; largest single
file `data/1.pdf` ≈ 48MB, within GitHub's 100MB hard limit). No Git LFS is
used.

## Verify after cloning

```bash
python3 validate_ingestion.py    # checks all 29 data-PDF sha256 hashes
python3 validate_ocr.py          # checks OCR artifacts against PDF hashes
python3 validate_canonical.py    # checks PDF hashes unchanged (check 12)
```

`corpus_inventory.json` holds the authoritative sha256 for every PDF; all
validation scripts fail loudly if any file is missing or modified.

## If PDFs are ever missing from a working copy

Restore them from Git rather than re-downloading:

```bash
git checkout HEAD -- data/           # data PDFs
git checkout HEAD -- .               # reference PDFs (root)
```

The processed artifacts under `data/processed/` are committed as well, so the
full pipeline state is available immediately after clone; the original PDFs
are only needed if you intend to re-run Parts 1–3 from scratch (not required
— every stage is hash-idempotent and its outputs are already present).
