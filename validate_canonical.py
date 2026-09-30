#!/usr/bin/env python3
"""
VectorRAG - Part 3 validation
=============================
Validates the canonical knowledge representation by reading ACTUAL FILES from
disk. Prints PASS/FAIL per check; exits non-zero on any failure.

Checks:
 1. All 29 documents represented
 2. All 501 pages represented
 3. Page numbers remain correct
 4. Every page retains raw_text
 5. Every page retains cleaned_text
 6. No raw OCR evidence deleted (raw_text byte-identical to Part 2 OCR text)
 7. Source metadata preserved (and consistent with corpus_inventory.json)
 8. Every detected section has a stable ID
 9. Section parent/child references are valid
10. No circular section hierarchy exists
11. Tables/lists have valid structure
12. Original PDF hashes are unchanged
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANON = ROOT / "data" / "processed" / "canonical"
OCR_DIR = ROOT / "data" / "processed" / "ocr"
INVENTORY = ROOT / "corpus_inventory.json"
INGEST_MANIFEST = ROOT / "data" / "processed" / "manifest.json"

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
    ing = json.loads((ROOT / "data/processed/manifest.json").read_text())
    num_pages = {d["document_id"]: d["num_pages"] for d in ing["documents"]}

    doc_records = {}
    jsonl = CANON / "documents.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                doc_records[d["document_id"]] = d

    # 1 - all 29 documents represented (page dirs + documents.jsonl records)
    dirs = sorted(p.name for p in CANON.iterdir() if p.is_dir())
    expected_dirs = sorted(f"doc_{i:03d}" for i in range(1, 30))
    check(1, "all 29 documents represented",
          dirs == expected_dirs and set(doc_records) == set(expected_dirs),
          f"{len(dirs)} page dirs, {len(doc_records)} document records")

    # 2-7 - per-page checks straight from disk
    total_pages = 0
    bad_pageno = missing_raw = missing_clean = raw_mismatch = 0
    missing_meta = []
    empty_clean_rep = 0
    for doc_id in expected_dirs:
        n = num_pages[doc_id]
        files = sorted((CANON / doc_id).glob("page_*.json"))
        total_pages += len(files)
        if len(files) != n:
            check(3, f"{doc_id} page count", False, f"{len(files)} != {n}")
            continue
        meta = mapping[str(int(doc_id[-3:]))]
        for pno, pf in enumerate(files, 1):
            d = json.loads(pf.read_text())
            if d["page_number"] != pno or d["document_id"] != doc_id:
                bad_pageno += 1
            if not isinstance(d.get("raw_text"), str) or not d["raw_text"]:
                missing_raw += 1
            if not isinstance(d.get("cleaned_text"), str):
                missing_clean += 1
            ocr = json.loads((OCR_DIR / doc_id / f"page_{pno:03d}.json").read_text())
            if d["raw_text"] != ocr["text"]:
                raw_mismatch += 1
            md = d.get("metadata", {})
            if not (md.get("document_title") == meta["title"]
                    and md.get("source_pdf") == meta["data_file"]
                    and md.get("reference_pdf") == meta["ref_file"]
                    and md.get("category") == meta["category"]
                    and md.get("m2c_relevance") == meta.get("m2c_relevance")
                    and md.get("source_url") == meta.get("url")
                    and md.get("authority") == 3):
                missing_meta.append(f"{doc_id}/p{pno}")
        # representative pages cleaned_text non-empty
        for pno in (1, max(1, n // 2), n):
            d = json.loads((CANON / doc_id / f"page_{pno:03d}.json").read_text())
            if len(d["cleaned_text"].strip()) < 20:
                empty_clean_rep += 1

    check(2, "all 501 pages represented", total_pages == 501, f"{total_pages} page files")
    check(3, "page numbers remain correct", bad_pageno == 0, f"{bad_pageno} mismatches")
    check(4, "every page retains raw_text", missing_raw == 0, f"{missing_raw} missing/empty")
    check(5, "every page retains cleaned_text", missing_clean == 0,
          f"{missing_clean} missing; representative pages with <20 chars: {empty_clean_rep}")
    check(6, "no raw OCR evidence deleted (raw_text == Part 2 OCR text, byte-identical)",
          raw_mismatch == 0, f"{raw_mismatch} mismatches out of 501")
    check(7, "source metadata preserved on every page", not missing_meta,
          f"{len(missing_meta)} pages with metadata mismatches")

    # 8-10 - section structure
    sec_re = re.compile(r"^doc_\d{3}_sec_\d{3}$")
    bad_ids, bad_parent, cycles = [], [], []
    for doc_id, rec in doc_records.items():
        seen = {}
        for s in rec["sections"]:
            if not sec_re.match(s["section_id"]) or s["section_id"] in seen:
                bad_ids.append(s["section_id"])
            seen[s["section_id"]] = s
        for s in rec["sections"]:
            pid = s["parent_section_id"]
            if pid is not None and pid not in seen:
                bad_parent.append(f"{doc_id}:{s['section_id']}->{pid}")
        for s in rec["sections"]:
            chain, cur, guard = set(), s["section_id"], 0
            while cur is not None and guard < 10000:
                if cur in chain:
                    cycles.append(f"{doc_id}:{s['section_id']}")
                    break
                chain.add(cur)
                nxt = seen.get(cur)
                cur = nxt["parent_section_id"] if nxt else None
                guard += 1
    n_sections = sum(len(r["sections"]) for r in doc_records.values())
    check(8, f"every detected section has a stable unique ID ({n_sections} sections)",
          not bad_ids, f"{len(bad_ids)} bad ids")
    check(9, "section parent/child references valid", not bad_parent,
          f"{len(bad_parent)} dangling parents")
    check(10, "no circular section hierarchy exists", not cycles, f"{len(cycles)} cycles")

    # 11 - tables/lists structure
    bad_tables = bad_lists = 0
    n_tables = n_lists = 0
    for doc_id in expected_dirs:
        for pf in (CANON / doc_id).glob("page_*.json"):
            d = json.loads(pf.read_text())
            for b in d["blocks"]:
                if b["type"] == "table":
                    n_tables += 1
                    if not (isinstance(b.get("rows"), list) and b["rows"]
                            and all(isinstance(r, list) for r in b["rows"])
                            and (b.get("headers") is None
                                 or isinstance(b.get("headers"), list))):
                        bad_tables += 1
                elif b["type"] == "list":
                    n_lists += 1
                    if not (isinstance(b.get("items"), list) and b["items"]
                            and isinstance(b.get("ordered"), bool)
                            and all(isinstance(i.get("text"), str) and i["text"].strip()
                                    for i in b["items"])):
                        bad_lists += 1
    check(11, f"tables/lists have valid structure ({n_tables} tables, {n_lists} lists)",
          not bad_tables and not bad_lists,
          f"{bad_tables} bad tables, {bad_lists} bad lists")

    # 12 - original PDF hashes unchanged
    changed = [mapping[str(i)]["data_file"] for i in range(1, 30)
               if sha256_file(ROOT / "data" / mapping[str(i)]["data_file"])
               != mapping[str(i)]["sha256"]]
    check(12, "original PDF hashes unchanged", not changed, f"changed: {changed or 'none'}")

    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print("  -", f)
        return 1
    print("VALIDATION PASSED: all checks green - canonical layer verified on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
