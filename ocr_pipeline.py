#!/usr/bin/env python3
"""
VectorRAG - Part 2: OCR / Text Recovery
=======================================
Produces real page-level OCR text for the 29 scanned data PDFs.

Engine: tesserocr 2.11.0 (bundled Tesseract 5.5.1, LSTM) with local language
data in ./tessdata (eng.traineddata + osd.traineddata). Pages are rendered as
complete pages at 300 DPI with PyMuPDF (never per-image-block OCR).

Idempotency: each page JSON records the hash of its source PDF + page number;
unchanged pages are skipped on rerun.

Usage:
    python3 ocr_pipeline.py --pilot      # docs 1, 6, 11 - sample pages only
    python3 ocr_pipeline.py [--workers N]  # all documents (default workers=2)
    python3 ocr_pipeline.py --check-only # report skip/without doing OCR
"""

import argparse
import json
import os
import hashlib
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import pymupdf
import tesserocr
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INVENTORY_PATH = ROOT / "corpus_inventory.json"
OCR_DIR = ROOT / "data" / "processed" / "ocr"
OCR_MANIFEST_PATH = ROOT / "data" / "processed" / "ocr_manifest.json"
OCR_ERRORS_PATH = ROOT / "data" / "processed" / "ocr_errors.json"
TESSDATA_DIR = ROOT / "tessdata"

ENGINE = f"tesserocr {tesserocr.tesseract_version().split()[1]} (Tesseract LSTM)"
LANGUAGE = "eng"
DPI = 300
OEM = tesserocr.OEM.LSTM_ONLY
PSM = tesserocr.PSM.AUTO
TESS_VARS = {"preserve_interword_spaces": "1"}
MAX_PIXELS = 40_000_000  # safety cap for oversized page renders

_API = None  # per-worker tesserocr API


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def page_hash(file_hash: str, page_number: int) -> str:
    return hashlib.sha256(f"{file_hash}:{page_number}".encode()).hexdigest()


def get_api():
    global _API
    if _API is None:
        _API = tesserocr.PyTessBaseAPI(
            path=str(TESSDATA_DIR), lang=LANGUAGE, oem=OEM, psm=PSM)
        for k, v in TESS_VARS.items():
            _API.SetVariable(k, v)
    return _API


def render_page(doc: "pymupdf.Document", page_number: int) -> Image.Image:
    """Render one complete PDF page (1-based) to a PIL image at OCR_DPI."""
    page = doc[page_number - 1]
    zoom = DPI / 72.0
    if page.rect.width * zoom * page.rect.height * zoom > MAX_PIXELS:
        zoom *= (MAX_PIXELS / (page.rect.width * zoom * page.rect.height * zoom)) ** 0.5
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def ocr_page(task) -> dict:
    """OCR a single page and atomically write its JSON artifact."""
    pdf_path, doc_id, page_number, fhash = task
    out_path = OCR_DIR / doc_id / f"page_{page_number:03d}.json"
    phash = page_hash(fhash, page_number)

    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("page_hash") == phash and prev.get("complete"):
                return {"doc": doc_id, "page": page_number, "status": "skipped",
                        "chars": prev.get("character_count", 0)}
        except Exception:
            pass  # corrupt/partial artifact -> redo

    t0 = time.time()
    doc = pymupdf.open(pdf_path)
    image = render_page(doc, page_number)
    doc.close()

    api = get_api()
    api.SetImage(image)
    text = api.GetUTF8Text()
    mean_conf = api.MeanTextConf()

    words = []
    it = api.GetIterator()
    if it is not None:
        level = tesserocr.RIL.WORD
        while True:
            try:
                w = it.GetUTF8Text(level)
                if w and w.strip():
                    x0, y0, x1, y1 = it.BoundingBox(level)
                    words.append([w, int(x0), int(y0), int(x1), int(y1),
                                  round(it.Confidence(level), 1)])
            except Exception:
                pass
            if not it.Next(level):
                break

    record = {
        "document_id": doc_id,
        "page_number": page_number,
        "source_pdf": Path(pdf_path).name,
        "page_hash": phash,
        "ocr_engine": ENGINE,
        "ocr_settings": {"oem": int(OEM), "psm": int(PSM), "dpi": DPI,
                         "language": LANGUAGE, "colorspace": "grayscale",
                         "variables": TESS_VARS},
        "render_size": [image.width, image.height],
        "text": text,
        "character_count": len(text),
        "word_count": len(words),
        "mean_confidence": mean_conf,
        "word_boxes": words,
        "processed_at": utcnow(),
        "duration_seconds": round(time.time() - t0, 2),
        "complete": True,
    }
    tmp = out_path.with_suffix(".json.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out_path)  # atomic: no partial artifacts on disk
    return {"doc": doc_id, "page": page_number, "status": "ok",
            "chars": len(text), "words": len(words), "conf": mean_conf,
            "secs": round(time.time() - t0, 2)}


def load_docs():
    inv = json.loads(INVENTORY_PATH.read_text())
    docs = []
    for num in sorted(inv["mapping"], key=int):
        meta = inv["mapping"][num]
        docs.append({
            "doc_id": f"doc_{int(num):03d}",
            "pdf_path": DATA_DIR / meta["data_file"],
            "file_hash": meta["sha256"],
            "meta": meta,
        })
    return docs


def pilot_pages(n_pages: int) -> list:
    """first, upper-quartile (content-dense region), middle, final page."""
    n = max(n_pages, 1)
    return sorted({1, max(1, n // 4), max(1, n // 2), n})


def scan_disk_results() -> dict:
    """Ground truth from files actually on disk."""
    results = {}
    if OCR_DIR.exists():
        for doc_dir in sorted(OCR_DIR.iterdir()):
            if not doc_dir.is_dir():
                continue
            pages = {}
            for pf in sorted(doc_dir.glob("page_*.json")):
                try:
                    pages[int(pf.stem.split("_")[1])] = json.loads(pf.read_text())
                except Exception:
                    pass
            results[doc_dir.name] = pages
    return results


def write_manifest(docs, started_at, mode: str):
    disk = scan_disk_results()
    per_doc, total_pages, pages_done, pages_failed, pages_with_text = [], 0, 0, 0, 0
    total_chars = total_words = 0
    confs = []
    for d in docs:
        expected = d["meta"]["pages"]
        pages = disk.get(d["doc_id"], {})
        ok = {n: p for n, p in pages.items() if p.get("complete")}
        chars = sum(p["character_count"] for p in ok.values())
        words = sum(p["word_count"] for p in ok.values())
        confs.extend(p["mean_confidence"] for p in ok.values() if p.get("word_count"))
        total_pages += expected
        pages_done += len(ok)
        pages_failed += sum(1 for p in pages.values() if not p.get("complete"))
        pages_with_text += sum(1 for p in ok.values() if p["character_count"] > 0)
        total_chars += chars
        total_words += words
        per_doc.append({
            "document_id": d["doc_id"], "source_pdf": d["meta"]["data_file"],
            "expected_pages": expected, "pages_ocr_completed": len(ok),
            "pages_with_text": sum(1 for p in ok.values() if p["character_count"] > 0),
            "total_characters": chars, "total_words": words,
            "mean_confidence": round(sum(p["mean_confidence"] for p in ok.values()) / len(ok), 1) if ok else None,
        })
    manifest = {
        "generated_at": utcnow(),
        "run_mode": mode,
        "ocr_engine": ENGINE,
        "processing_settings": {"oem": int(OEM), "psm": int(PSM), "dpi": DPI,
                                "language": LANGUAGE, "colorspace": "grayscale",
                                "variables": TESS_VARS,
                                "tessdata_dir": str(TESSDATA_DIR),
                                "render": "full page via PyMuPDF (never per-image-block)"},
        "started_at": started_at, "finished_at": utcnow(),
        "total_documents": len(docs),
        "total_pages": total_pages,
        "pages_processed": pages_done,
        "pages_failed": pages_failed,
        "pages_with_text": pages_with_text,
        "total_characters": total_chars,
        "total_words": total_words,
        "mean_confidence": round(sum(confs) / len(confs), 1) if confs else None,
        "documents": per_doc,
    }
    OCR_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    docs = load_docs()
    started_at = utcnow()

    tasks = []
    for d in docs:
        if not d["pdf_path"].exists():
            print(f"[FAIL] missing {d['pdf_path']}", flush=True)
            continue
        n = d["meta"]["pages"]
        if args.pilot and d["doc_id"] not in {"doc_001", "doc_006", "doc_011"}:
            continue
        pages = pilot_pages(n) if args.pilot else range(1, n + 1)
        for pno in pages:
            tasks.append((str(d["pdf_path"]), d["doc_id"], pno, d["file_hash"]))

    print(f"OCR pipeline: mode={'pilot' if args.pilot else 'full'} "
          f"docs={len({t[1] for t in tasks})} pages={len(tasks)} "
          f"engine={ENGINE}", flush=True)

    if args.check_only:
        done = sum(1 for t in tasks if (OCR_DIR / t[1] / f"page_{t[2]:03d}.json").exists())
        print(f"CHECK_ONLY: would skip={done} would process={len(tasks) - done}", flush=True)
        return 0

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    failures, done_count, t_start = [], 0, time.time()
    current_doc = {}

    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(ocr_page, tasks, chunksize=1):
            done_count += 1
            if res["doc"] != current_doc.get("doc"):
                if current_doc:
                    print(f"DOC_COMPLETE {current_doc['doc']}: {current_doc['ok']}/{current_doc['total']} pages", flush=True)
                current_doc = {"doc": res["doc"], "ok": 0, "total": 0}
            current_doc["total"] += 1
            if res["status"] == "skipped":
                current_doc["ok"] += 1
            elif res["status"] == "ok":
                current_doc["ok"] += 1
                if done_count % 25 == 0 or res["doc"] in {"doc_001", "doc_006", "doc_011"}:
                    print(f"  [{done_count}/{len(tasks)}] {res['doc']} p{res['page']:>3}: "
                          f"{res['chars']:>5} chars, {res['words']:>4} words, conf={res['conf']}, {res['secs']}s", flush=True)
            else:
                failures.append(res)
        if current_doc:
            print(f"DOC_COMPLETE {current_doc['doc']}: {current_doc['ok']}/{current_doc['total']} pages", flush=True)

    if failures:
        OCR_ERRORS_PATH.write_text(json.dumps(failures, indent=1))
    print(f"OCR loop finished: processed_or_skipped={done_count} failures={len(failures)} "
          f"elapsed={round(time.time() - t_start, 1)}s", flush=True)

    manifest = write_manifest(docs, started_at, "pilot" if args.pilot else "full")
    print(f"ocr_manifest.json written: pages_processed={manifest['pages_processed']}/"
          f"{manifest['total_pages']} failed={manifest['pages_failed']} "
          f"chars={manifest['total_characters']} words={manifest['total_words']} "
          f"mean_conf={manifest['mean_confidence']}", flush=True)
    print("ALL_DONE", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
