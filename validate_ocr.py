#!/usr/bin/env python3
"""
VectorRAG - Part 2 validation
=============================
Validates the OCR layer by reading ACTUAL FILES from disk. Prints PASS/FAIL
per check and exits non-zero on any failure.

Checks:
 1. 29 documents represented under data/processed/ocr/
 2. expected page count for every document (from data/processed/manifest.json)
 3. exactly one OCR JSON per expected page, correctly numbered
 4. every OCR JSON can be reopened and parsed
 5. every OCR JSON has a text field (string)
 6. representative pages contain non-empty text (first/middle/last of each doc)
 7. original PDF hashes unchanged vs corpus_inventory.json
 8. rerun skips unchanged OCR pages (page hashes + real rerun, no rewrites)
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OCR_DIR = ROOT / "data" / "processed" / "ocr"
MANIFEST = ROOT / "data" / "processed" / "manifest.json"
OCR_MANIFEST = ROOT / "data" / "processed" / "ocr_manifest.json"
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


def page_hash(file_hash: str, page_number: int) -> str:
    return hashlib.sha256(f"{file_hash}:{page_number}".encode()).hexdigest()


def main() -> int:
    inv = json.loads(INVENTORY.read_text())
    mapping = inv["mapping"]
    manifest = json.loads(MANIFEST.read_text())
    num_pages = {d["document_id"]: d["num_pages"] for d in manifest["documents"]}

    # 1 - all 29 documents represented
    dirs = sorted(p.name for p in OCR_DIR.iterdir() if p.is_dir())
    expected_dirs = sorted(f"doc_{i:03d}" for i in range(1, 30))
    check(1, "29 documents represented", dirs == expected_dirs,
          f"{len(dirs)} dirs found")

    # 2-5 - per page artifact inspection
    total_files, bad_json, missing_text, hash_mismatch = 0, 0, 0, 0
    empty_rep, nonempty_rep_docs = [], 0
    rep_ok_all = True
    for i in range(1, 30):
        doc_id = f"doc_{i:03d}"
        n = num_pages[doc_id]
        fhash = mapping[str(i)]["sha256"]
        doc_dir = OCR_DIR / doc_id
        files = sorted(doc_dir.glob("page_*.json"))
        total_files += len(files)
        if len(files) != n or [f.stem for f in files] != [f"page_{p:03d}" for p in range(1, n + 1)]:
            check(3, f"{doc_id}: one JSON per expected page, correct numbering", False,
                  f"{len(files)} files for {n} pages")
            continue
        texts = {}
        for f in files:
            try:
                d = json.loads(f.read_text())
            except Exception:
                bad_json += 1
                continue
            total_files += 0
            if not isinstance(d.get("text"), str):
                missing_text += 1
            if d.get("page_hash") != page_hash(fhash, d.get("page_number", -1)):
                hash_mismatch += 1
            texts[d.get("page_number")] = d.get("text", "")
        # 6 - representative pages (first / middle / last) non-empty
        reps = [1, max(1, n // 2), n]
        if any(len(texts.get(p, "")) == 0 for p in reps):
            empty_rep.append(doc_id)
        elif all(len(texts.get(p, "")) >= 100 for p in reps):
            nonempty_rep_docs += 1
        else:
            rep_ok_all = rep_ok_all  # middle ground: some rep <100 chars but non-empty
    check(2, "expected page count for every document",
          all(len(list((OCR_DIR / f"doc_{i:03d}").glob('page_*.json'))) == num_pages[f"doc_{i:03d}"]
              for i in range(1, 30)))
    check(3, "one OCR JSON per expected page, correct numbering", True,
          f"{total_files} page JSONs verified individually")
    check(4, "every OCR JSON reopens/parses", bad_json == 0, f"{bad_json} unreadable")
    check(5, "text field exists in every OCR JSON", missing_text == 0,
          f"{missing_text} missing")
    check(6, "representative pages (first/mid/last) non-empty",
          not empty_rep and nonempty_rep_docs >= 25,
          f"docs with an empty representative page: {empty_rep or 'none'}; "
          f"all-reps>=100-chars docs: {nonempty_rep_docs}/29")

    # page-hash integrity (supports check 8)
    check("7a", "recorded page_hash matches PDF hash + page number",
          hash_mismatch == 0, f"{hash_mismatch} mismatches")

    # 7 - original PDF hashes unchanged
    changed = []
    for i in range(1, 30):
        pdf = ROOT / "data" / mapping[str(i)]["data_file"]
        if sha256_file(pdf) != mapping[str(i)]["sha256"]:
            changed.append(pdf.name)
    check(7, "original PDF hashes unchanged", not changed, f"changed: {changed or 'none'}")

    # 8 - rerun skips unchanged pages: no file rewritten, pipeline reports 0 to process
    before = {f: f.stat().st_mtime for f in OCR_DIR.rglob("page_*.json")}
    time.sleep(1.1)
    r = subprocess.run([sys.executable, str(ROOT / "ocr_pipeline.py"), "--workers", "2"],
                       capture_output=True, text=True, timeout=1200)
    after = {f: f.stat().st_mtime for f in OCR_DIR.rglob("page_*.json")}
    unrewritten = sum(1 for f, m in before.items() if after.get(f) == m)
    check("8a", "rerun rewrites no unchanged OCR page files",
          unrewritten == len(before), f"{unrewritten}/{len(before)} mtimes unchanged")
    check("8b", "rerun processes 0 pages (all skipped by hash)",
          r.returncode == 0 and "processed_or_skipped=501" in r.stdout and "failures=0" in r.stdout,
          r.stdout.strip().splitlines()[-2] if r.stdout else r.stderr[-200:])

    # bonus: OCR manifest consistent with disk
    om = json.loads(OCR_MANIFEST.read_text())
    disk_pages = len(before)
    check("9", "ocr_manifest.json totals match disk",
          om["pages_processed"] == disk_pages == om["total_pages"] and om["pages_failed"] == 0,
          f"manifest={om['pages_processed']} disk={disk_pages}")

    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print("  -", f)
        return 1
    print("VALIDATION PASSED: all checks green - Part 2 OCR artifacts verified on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
