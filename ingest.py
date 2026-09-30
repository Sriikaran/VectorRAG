#!/usr/bin/env python3
"""
VectorRAG - Part 1: Corpus Ingestion Layer
==========================================
Ingests the 29 scanned data PDFs (data/1.pdf .. data/29.pdf) using PyMuPDF,
preserving the structure that is extractable from image-only pages (page
geometry, block layout, reading order) and enriching every document with the
metadata curated in corpus_inventory.json (title, category, M2C relevance,
source URL, authority).

Outputs (idempotent - unchanged PDFs are skipped by hash):
    data/processed/documents/doc_001.json .. doc_029.json
    data/processed/manifest.json
    data/processed/parse_errors.json

Docling is the documented primary extractor but requires network model
downloads that are blocked in this environment; PyMuPDF is the fallback and is
used for all documents here (recorded per document as extraction_method).
"""

import json
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pymupdf  # PyMuPDF

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INVENTORY_PATH = ROOT / "corpus_inventory.json"
DOCUMENTS_DIR = ROOT / "data" / "processed" / "documents"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.json"
ERRORS_PATH = ROOT / "data" / "processed" / "parse_errors.json"

AUTHORITY = "SAP Help Portal - official SAP S/4HANA product documentation (Utilities)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_document(pdf_path: Path, meta: dict, doc_id: str) -> dict:
    """Extract page/structure information from one scanned PDF via PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    pages_out = []
    for pno in range(len(doc)):
        page = doc[pno]
        raw = page.get_text("dict")
        blocks = []
        for idx, blk in enumerate(raw.get("blocks", [])):
            entry = {
                "block_index": idx,
                "block_type": "text" if blk.get("type") == 0 else "image",
                "bbox": [round(v, 2) for v in blk.get("bbox", [])],
            }
            if blk.get("type") == 1:
                entry["image_width"] = blk.get("width")
                entry["image_height"] = blk.get("height")
            blocks.append(entry)
        pages_out.append({
            "page_number": pno + 1,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "text_characters": len(page.get_text().strip()),
            "blocks": blocks,
        })
    doc.close()

    # Structure extractable from a pure image scan: reading order of layout
    # blocks. Headings/paragraphs/lists/tables are recovered in Part 2 (OCR).
    structure = {
        "note": "Source pages are image-only scans; text-level structure "
                "(headings, paragraphs, lists, tables) is recovered by the "
                "Part 2 OCR layer.",
        "reading_order": [
            {"page_number": p["page_number"],
             "block_sequence": [b["block_index"] for b in p["blocks"]]}
            for p in pages_out
        ],
        "headings": [],
        "paragraphs": [],
        "lists": [],
        "tables": [],
    }

    return {
        "document_id": doc_id,
        "original_filename": meta["data_file"],
        "source_reference": meta["ref_file"],
        "title": meta["title"],
        "category": meta["category"],
        "m2c_relevance": meta["m2c_relevance"],
        "source_url": meta.get("source_url", ""),
        "authority": AUTHORITY,
        "file_hash": meta["sha256"],
        "file_size": meta["size_bytes"],
        "extraction_method": "pymupdf",
        "num_pages": len(pages_out),
        "pages": pages_out,
        "structure": structure,
        "processed_at": utcnow(),
    }


def main() -> int:
    inventory = json.loads(INVENTORY_PATH.read_text())
    mapping = inventory["mapping"]

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    old_manifest = {}
    if MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text())
            old_manifest = {d["document_id"]: d for d in old_manifest.get("documents", [])}
        except Exception:
            old_manifest = {}

    manifest_docs, errors, processed, skipped = [], [], 0, 0

    for num in sorted(mapping, key=int):
        doc_id = f"doc_{int(num):03d}"
        meta = dict(mapping[num])
        meta["m2c_relevance"] = meta.get("m2c_relevance", meta.get("m2c", ""))
        meta["source_url"] = meta.get("url", "")
        pdf_path = DATA_DIR / meta["data_file"]

        if not pdf_path.exists():
            errors.append({"document_id": doc_id, "error": f"missing file: {pdf_path}"})
            continue

        actual_hash = sha256_file(pdf_path)
        if actual_hash != meta["sha256"]:
            errors.append({"document_id": doc_id,
                           "error": f"hash mismatch: disk {actual_hash} != inventory {meta['sha256']}"})
            continue

        prev = old_manifest.get(doc_id)
        out_path = DOCUMENTS_DIR / f"{doc_id}.json"
        if (prev and prev.get("file_hash") == actual_hash
                and prev.get("status") == "ok" and out_path.exists()):
            try:
                json.loads(out_path.read_text())  # sanity: reopenable
                skipped += 1
                manifest_docs.append(prev)
                print(f"[skip] {doc_id} unchanged (hash {actual_hash[:12]}...)")
                continue
            except Exception:
                pass  # corrupt output -> reprocess

        t0 = time.time()
        try:
            record = extract_document(pdf_path, meta, doc_id)
            out_path.write_text(json.dumps(record, indent=1))
            entry = {
                "document_id": doc_id,
                "original_filename": meta["data_file"],
                "source_reference": meta["ref_file"],
                "title": meta["title"],
                "category": meta["category"],
                "num_pages": record["num_pages"],
                "file_hash": actual_hash,
                "file_size": meta["size_bytes"],
                "extraction_method": "pymupdf",
                "status": "ok",
                "document_json_path": str(out_path.relative_to(ROOT)),
                "processed_at": record["processed_at"],
                "duration_seconds": round(time.time() - t0, 2),
            }
            manifest_docs.append(entry)
            processed += 1
            print(f"[ok]   {doc_id}: {meta['data_file']} -> {record['num_pages']} pages "
                  f"({entry['duration_seconds']}s)")
        except Exception as exc:
            errors.append({"document_id": doc_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {doc_id}: {exc}")

    manifest = {
        "generated_at": utcnow(),
        "total_documents": len(manifest_docs),
        "documents": sorted(manifest_docs, key=lambda d: d["document_id"]),
    }
    # idempotency: when nothing changed, leave existing manifest/errors untouched
    if processed or errors or not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
        ERRORS_PATH.write_text(json.dumps(errors, indent=1))

    print(f"\nPart 1 ingestion: processed={processed} skipped={skipped} "
          f"errors={len(errors)} manifest={MANIFEST_PATH}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
