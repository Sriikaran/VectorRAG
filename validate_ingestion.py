#!/usr/bin/env python3
"""
VectorRAG - Part 1 validation (ingestion layer)
===============================================
Validates Part 1 outputs by reading ACTUAL FILES from disk:
 1. All 29 documents ingested (doc_001..doc_029 JSONs exist and reopen)
 2. Page counts match the source PDFs
 3. Manifest is consistent (29 entries, hashes match corpus_inventory.json)
 4. parse_errors.json exists and is a valid JSON list
 5. Required metadata present on every document (title, category,
    m2c_relevance, source_url, authority, extraction_method, hashes)
 6. Original PDF hashes unchanged vs corpus_inventory.json
 7. Idempotent rerun: re-running ingest.py rewrites nothing
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "data" / "processed" / "documents"
MANIFEST = ROOT / "data" / "processed" / "manifest.json"
ERRORS = ROOT / "data" / "processed" / "parse_errors.json"
INVENTORY = ROOT / "corpus_inventory.json"

failures = []


def check(num, name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append((num, name, detail))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    inv = json.loads(INVENTORY.read_text())
    mapping = inv["mapping"]

    # 1 - all 29 documents present and reopenable
    bad = []
    for i in range(1, 30):
        p = DOCS / f"doc_{i:03d}.json"
        try:
            json.loads(p.read_text())
        except Exception as e:
            bad.append(f"doc_{i:03d}: {e}")
    check(1, "all 29 document JSONs exist and reopen", not bad, f"{bad or 'none'}")

    # 2 - page counts match source PDFs (via manifest) and documents
    man = json.loads(MANIFEST.read_text())
    entries = {d["document_id"]: d for d in man["documents"]}
    bad_pages = [d["document_id"] for i in range(1, 30)
                 for d in [entries.get(f"doc_{i:03d}", {})]
                 if d.get("num_pages") != mapping[str(i)]["pages"]]
    check(2, "manifest page counts match source PDFs", not bad_pages,
          f"{len(bad_pages)} mismatches")

    # 3 - manifest consistency (hashes match inventory)
    bad_hash = [d["document_id"] for i in range(1, 30)
                for d in [entries.get(f"doc_{i:03d}", {})]
                if d.get("file_hash") != mapping[str(i)]["sha256"]]
    check(3, "manifest file hashes match corpus_inventory.json", not bad_hash,
          f"{len(bad_hash)} mismatches")

    # 4 - parse_errors.json valid
    try:
        errs = json.loads(ERRORS.read_text())
        ok4 = isinstance(errs, list)
    except Exception:
        errs, ok4 = None, False
    check(4, "parse_errors.json exists and is valid", ok4,
          f"{len(errs) if isinstance(errs, list) else 'n/a'} errors recorded")

    # 5 - required metadata on every document
    bad_meta = []
    required = ("title", "category", "m2c_relevance", "source_url", "authority",
                "file_hash", "extraction_method", "num_pages", "pages", "structure")
    for i in range(1, 30):
        d = json.loads((DOCS / f"doc_{i:03d}.json").read_text())
        if any(not d.get(k) and k != "source_url" for k in required) \
                or d.get("extraction_method") != "pymupdf" \
                or len(d.get("pages", [])) != d.get("num_pages"):
            bad_meta.append(d.get("document_id", f"doc_{i:03d}"))
    check(5, "required metadata present on every document", not bad_meta,
          f"{len(bad_meta)} bad")

    # 6 - original PDF hashes unchanged
    changed = [mapping[str(i)]["data_file"] for i in range(1, 30)
               if sha256_file(ROOT / "data" / mapping[str(i)]["data_file"])
               != mapping[str(i)]["sha256"]]
    check(6, "original PDF hashes unchanged", not changed, f"changed: {changed or 'none'}")

    # 7 - idempotent rerun rewrites nothing
    before = {p: p.stat().st_mtime for p in DOCS.glob("doc_*.json")}
    before_m = MANIFEST.read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "ingest.py")],
                       capture_output=True, text=True, timeout=600)
    after = {p: p.stat().st_mtime for p in DOCS.glob("doc_*.json")}
    unchanged = sum(1 for p, m in before.items() if after.get(p) == m)
    check(7, "idempotent rerun rewrites nothing",
          r.returncode == 0 and unchanged == len(before)
          and MANIFEST.read_bytes() == before_m and "skipped=29" in r.stdout,
          f"{unchanged}/{len(before)} mtimes unchanged")

    print()
    if failures:
        print(f"INGESTION VALIDATION FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print("  -", f)
        return 1
    print("INGESTION VALIDATION PASSED: Part 1 artifacts verified on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
