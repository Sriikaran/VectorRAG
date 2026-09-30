#!/usr/bin/env python3
"""
VectorRAG - Part 3: Cleaning + Canonical Knowledge Representation
================================================================
Transforms the Part 2 page-level OCR into a canonical knowledge representation
for the later chunking stage. NO content is rewritten, summarized, paraphrased
or invented: raw OCR text is preserved verbatim as evidence (raw_text), and
cleaning is restricted to deterministic mechanical extraction artifacts
(whitespace, line-wrapping, hyphenation breaks at line ends, exact duplicate
lines, marked navigation chrome).

Detection signals (all local, deterministic):
  - word bounding boxes (line reconstruction, indentation, column clustering)
  - page renders at 150 dpi: median stroke-width ratio (boldness), ink ratio,
    background brightness (graphic regions)
  - repeated patterns across pages (print-timestamp chrome)

Outputs (OCR artifacts are never modified):
    data/processed/canonical/doc_XXX/page_XXX.json   canonical page schema
    data/processed/canonical/documents.jsonl         document -> sections -> pages -> blocks
    data/processed/canonical/canonical_manifest.json run statistics

Idempotency: a page is reprocessed only if its source page hash or the
pipeline/schema version changed. --force reprocesses everything.

Authority levels:
    3 = SAP Help Portal source content (raw/cleaned text, blocks)
    2 = curated reference metadata (title, category, m2c_relevance, source_url)
    1 = generated/internal metadata (ids, detection decisions, thresholds)
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent
OCR_DIR = ROOT / "data" / "processed" / "ocr"
INVENTORY_PATH = ROOT / "corpus_inventory.json"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.json"
CANON_DIR = ROOT / "data" / "processed" / "canonical"
PAGES_DIR = CANON_DIR
DOC_JSONL = CANON_DIR / "documents.jsonl"
CANON_MANIFEST = CANON_DIR / "canonical_manifest.json"

PIPELINE_VERSION = "1.1"
SCHEMA_VERSION = "1.0"

# --- detection settings (calibrated on rendered 300 dpi pages, OCR coords) ---
CONFIG = {
    "render_dpi": 150,
    "render_scale": 0.5,            # 150dpi coords = 300dpi OCR coords * 0.5
    "dark_threshold": 160,          # gray < threshold counts as ink
    "stroke_bold_ratio": 1.25,      # med_stroke >= body_median * 1.25 => bold
    "stroke_bold_abs": 0.128,       # absolute floor for boldness
    "heading_min_bg": 253,          # headings must sit on (near-)white background
    "level2_min_height": 58,        # chapter heading height (300dpi px); 56-57px
                                    # bold lines are same-level section headings
    "level2_min_stroke": 0.10,
    "level3_min_height": 40,        # bold h>=40 => level 3, bold h<40 => level 4
    "heading_max_len": 95,
    "column_gap": 60,               # x-gap (300dpi px) separating table cells
    "column_align_tol": 28,         # column start alignment tolerance
    "graphic_bg": 252,              # median background < this => graphic region
    "cut_off_y": 3260,              # line starting below this => cut-off fragment
    "cut_off_max_h": 26,
    "low_word_conf": 60,
    "header_ink_factor": 1.25,      # first-row ink vs rest => header row
    "para_gap_factor": 1.6,         # gap > pitch * factor => block boundary
    "heading_gap_factor": 1.3,      # layout-only heading candidate gaps
    "nav_timestamp_y": 150,         # chrome band
}

RE_TS = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*$")
RE_NOTE = re.compile(r"^(?:([^\sa-z]{1,4})\s+)?(Note|Caution|Warning|Recommendation|Tip|Example)\s*:?\s*$")
RE_BULLET = re.compile(r"^([eo\u2022\u25cf\u00b7*\u2023\u25e6\u2013-])\s+([A-Z0-9(\"'\u201c]).+")
RE_ORDERED = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")
RE_URLCONT = re.compile(r"^[\w&%?./=#:\-~]+$")
RE_COVER_META = re.compile(r"^(Generated on:|SAP S/4HANA|Public$|Original content:)")
RE_HYPHEN = re.compile(r"(\S)-$")

_API = None  # placeholder (not used; kept for worker init symmetry)
_IMG_CACHE = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- geometry --

def reconstruct_lines(page: dict) -> list:
    """Align word_boxes to OCR text lines. Returns [{text, words, bbox, ...}]."""
    words = page.get("word_boxes") or []
    out, wi, ok = [], 0, True
    for raw in page["text"].split("\n"):
        toks = raw.split()
        chunk = words[wi:wi + len(toks)]
        wi += len(toks)
        if len(chunk) != len(toks):
            ok = False
            out.append({"text": raw, "words": [], "bbox": None, "aligned": False})
            continue
        if not toks:
            out.append({"text": raw, "words": [], "bbox": None, "aligned": True})
            continue
        bbox = (min(w[1] for w in chunk), min(w[2] for w in chunk),
                max(w[3] for w in chunk), max(w[4] for w in chunk))
        out.append({"text": raw, "words": chunk, "bbox": bbox, "aligned": True})
    for ln in out:
        ln["aligned"] = ok and ln.get("aligned", False)
    return out


def line_cells(line: dict, gap: int) -> list:
    """Cluster a line's words into column cells by x-gaps (reading order)."""
    words = sorted(line["words"], key=lambda w: w[1])
    if not words:
        return []
    cells = [[words[0]]]
    for w in words[1:]:
        if w[1] - cells[-1][-1][3] > gap:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


# ------------------------------------------------------------- pixel stats --

def get_render(pdf_path: str, page_number: int):
    key = (pdf_path, page_number)
    if key not in _IMG_CACHE:
        doc = pymupdf.open(pdf_path)
        dpi = CONFIG["render_dpi"]
        pix = doc[page_number - 1].get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72, dpi / 72), colorspace=pymupdf.csGRAY)
        _IMG_CACHE.clear()  # one page cached per worker at a time
        _IMG_CACHE[key] = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        doc.close()
    return _IMG_CACHE[key]


def stroke_stats(img: Image.Image, bbox, scale: float) -> dict:
    """Median horizontal ink run-length / line height + ink ratio + background."""
    x0, y0, x1, y1 = [int(v * scale) for v in bbox]
    x0 = max(0, x0 - 2); y0 = max(0, y0 - 2)
    x1 = min(img.width, x1 + 2); y1 = min(img.height, y1 + 2)
    crop = img.crop((x0, y0, x1, y1))
    W, H = crop.size
    data = crop.tobytes()
    thr = CONFIG["dark_threshold"]
    runs, dark, bg_vals = [], 0, []
    for row in range(H):
        base = row * W
        run = 0
        for col in range(W):
            p = data[base + col]
            if p < thr:
                run += 1; dark += 1
            else:
                if run: runs.append(run); run = 0
                if col < 3 or col >= W - 3: bg_vals.append(p)
        if run: runs.append(run)
    area = max(1, W * H)
    line_h = max(1.0, (bbox[3] - bbox[1]) * scale)
    return {
        "med_stroke": (statistics.median(runs) / line_h) if runs else 0.0,
        "ink": dark / area,
        "bg": statistics.median(bg_vals) if bg_vals else 255,
    }


# ------------------------------------------------------------ text cleaning --

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    return re.sub(r" {2,}", " ", s).strip()


def join_wrapped(lines: list) -> str:
    """Join wrapped paragraph lines; remove line-break hyphenation artifacts
    and exact duplicated lines caused by extraction."""
    out = ""
    for ln in lines:
        t = clean_text(ln)
        if not t or t == out:
            continue
        if not out:
            out = t
            continue
        m = RE_HYPHEN.search(out)
        if m and t[:1].islower():
            out = out + t  # line-break hyphenation: keep hyphen, drop the break
        else:
            out += " " + t
    return out


# ---------------------------------------------------------- page processing --

def classify_lines(page: dict, pdf_path: str) -> dict:
    """Reconstruct lines, compute geometry/pixel stats, assign tentative roles."""
    lines = reconstruct_lines(page)
    scale = CONFIG["render_scale"]
    need_px = any(ln["bbox"] for ln in lines)
    img = get_render(pdf_path, page["page_number"]) if need_px else None

    body_strokes = []
    for ln in lines:
        if not ln["text"].strip():
            ln["stats"] = None
            continue
        if ln["bbox"] is None:
            ln["stats"] = {"med_stroke": 0.0, "ink": 0.0, "bg": 255}
            continue
        st = stroke_stats(img, ln["bbox"], scale)
        ln["stats"] = st
        toks = ln["text"].split()
        if (40 <= ln["bbox"][3] - ln["bbox"][1] <= 46 and len(toks) >= 8
                and st["bg"] >= CONFIG["graphic_bg"]):
            body_strokes.append(st["med_stroke"])
    body_median = statistics.median(body_strokes) if body_strokes else 0.108

    # line pitch: prefer the tight text-line cluster (small deltas); bullet
    # leading / section gaps must not pollute the median
    tops = [ln["bbox"][1] for ln in lines if ln["bbox"]]
    deltas = [b - a for a, b in zip(tops, tops[1:]) if 20 < b - a < 150]
    small = [d for d in deltas if d < 80]
    pitch = statistics.median(small) if small else (
        statistics.median(deltas) if deltas else 63.0)

    for i, ln in enumerate(lines):
        ln["index"] = i
        s = ln["text"].strip()
        ln["s"] = s
        ln["cells"] = line_cells(ln, CONFIG["column_gap"]) if ln["words"] else []
        ln["role"] = None
        if not s:
            continue
        st = ln["stats"]
        bbox = ln["bbox"]
        h = (bbox[3] - bbox[1]) if bbox else 0
        n_words = len(ln["words"])
        conf = (sum(w[5] for w in ln["words"]) / n_words) if n_words else 0.0
        ln["h"], ln["conf"] = h, conf

        # 1) navigation chrome: print timestamp in the top band
        if RE_TS.match(s) and bbox and bbox[1] < CONFIG["nav_timestamp_y"]:
            ln["role"] = "nav_chrome"
            continue
        # 2) cover-page document title (very large font) - before heading rules
        if page["page_number"] == 1 and bbox and h >= 80:
            ln["role"] = "document_title"
            continue
        # 3) note/labels (also covers cover-page 'Warning')
        m = RE_NOTE.match(s)
        if m:
            ln["role"] = "label_" + m.group(2).lower()
            ln["icon_glyph"] = m.group(1)
            continue
        # cover pages carry no chapter headings: everything below is meta/body
        if page["page_number"] == 1:
            ln["role"] = "paragraph"
            continue
        # 3) list items
        mb = RE_BULLET.match(s)
        mo = RE_ORDERED.match(s)
        if mb and len(ln["cells"]) == 1:
            ln["role"] = "li_unordered"
            ln["marker"] = mb.group(1)
            ln["item_text"] = s[mb.end(1):].strip()
            continue
        if mo and len(ln["cells"]) == 1:
            ln["role"] = "li_ordered"
            ln["marker"] = mo.group(1)
            ln["item_text"] = mo.group(2).strip()
            continue
        # 4) cut-off fragment at page bottom (before heading rules: cut lines
        #    can OCR bold-looking and must never become headings)
        if bbox and (bbox[1] > CONFIG["cut_off_y"] and h <= CONFIG["cut_off_max_h"]):
            ln["role"] = "fragment_cut"
            continue
        # 5) headings (single-cluster, white background, short-ish, real height,
        #    no menu-path glyphs, sane text)
        if (len(ln["cells"]) == 1 and ln["aligned"] and bbox
                and h >= 24
                and st["bg"] >= CONFIG["heading_min_bg"]
                and conf >= 68
                and (s[0].isupper() or s[0].isdigit())
                and "|" not in s and "\u00bb" not in s
                and not s.endswith(".")
                and len(s) <= CONFIG["heading_max_len"]):
            if (h >= CONFIG["level2_min_height"]
                    and st["med_stroke"] >= CONFIG["level2_min_stroke"]
                    and not s.endswith(".")):
                ln["role"] = "heading"; ln["level"] = 2; ln["heading_conf"] = "high"
                continue
            bold = st["med_stroke"] >= max(CONFIG["stroke_bold_abs"],
                                           body_median * CONFIG["stroke_bold_ratio"])
            if bold and CONFIG["level3_min_height"] <= h < CONFIG["level2_min_height"]:
                ln["role"] = "heading"; ln["level"] = 3; ln["heading_conf"] = "high"
                continue
            if bold and h < CONFIG["level3_min_height"]:
                ln["role"] = "heading"; ln["level"] = 4; ln["heading_conf"] = "high"
                continue
            # layout-only candidate: isolated short line with clear gaps
            prev_b = next((lines[j]["bbox"] for j in range(i - 1, -1, -1)
                           if lines[j]["bbox"] and lines[j]["text"].strip()), None)
            next_b = next((lines[j]["bbox"] for j in range(i + 1, len(lines))
                           if lines[j]["bbox"] and lines[j]["text"].strip()), None)
            if (h >= 26 and s[0].isupper() and not s.endswith((".", ",", ";", ":"))
                    and len(ln["cells"]) == 1):
                gb = (bbox[1] - prev_b[3]) if prev_b else None
                ga = (next_b[1] - bbox[3]) if next_b else None
                need = CONFIG["heading_gap_factor"] * pitch
                if (gb is None or gb >= need) and (ga is not None and ga >= need):
                    ln["role"] = "heading"
                    ln["level"] = 3 if h >= CONFIG["level3_min_height"] else 4
                    ln["heading_conf"] = "low"
                    continue
        # 6) fragments handled above; default paragraph/table
        ln["role"] = "table_row" if len(ln["cells"]) >= 2 else "paragraph"

    return {"lines": lines, "pitch": pitch, "body_median": body_median}


def group_blocks(cls: dict, page: dict, meta: dict) -> list:
    """Group classified lines into ordered canonical blocks.

    Blank OCR text lines are dropped before grouping: Tesseract inserts them
    inconsistently (even inside wrapped paragraphs), and all grouping
    decisions here are geometry-based. Blank lines remain in raw_text.
    """
    lines = [ln for ln in cls["lines"] if ln["s"]]
    pitch = cls["pitch"]
    doc_id, pno = meta["document_id"], page["page_number"]
    blocks, noise = [], []

    def mk(bidx, btype, lns, **extra):
        words = [w for ln in lns for w in ln["words"]]
        bbox = None
        if lns and all(ln["bbox"] for ln in lns):
            bbox = [min(ln["bbox"][0] for ln in lns), min(ln["bbox"][1] for ln in lns),
                    max(ln["bbox"][2] for ln in lns), max(ln["bbox"][3] for ln in lns)]
        conf = round(sum(w[5] for w in words) / len(words), 1) if words else None
        low = sum(1 for w in words if w[5] < CONFIG["low_word_conf"])
        b = {"block_id": f"{doc_id}_p{pno:03d}_b{bidx:03d}", "type": btype,
             "page_number": pno, "bbox": [round(v, 1) for v in bbox] if bbox else None,
             "ocr_confidence": conf, "flags": [], **extra}
        if low:
            b["flags"].append(f"low_confidence_words:{low}")
        if any(ln.get("stats") and ln["stats"]["bg"] < CONFIG["graphic_bg"] for ln in lns):
            b["flags"].append("in_graphic_region")
        return b

    i = 0
    bidx = 0
    while i < len(lines):
        ln = lines[i]
        if not ln["s"]:
            i += 1
            continue
        role = ln["role"]

        if role == "nav_chrome":
            noise.append({"type": "nav_chrome", "subtype": "print_timestamp",
                          "text": ln["s"], "bbox": [round(v, 1) for v in ln["bbox"]],
                          "reason": "browser/print timestamp repeated on every page"})
            i += 1
            continue

        if role and role.startswith("label_"):
            label = role.split("_", 1)[1]
            lns, j = [ln], i + 1
            while j < len(lines):
                nx = lines[j]
                if not nx["s"] or nx["role"] != "paragraph" or not nx["bbox"] or not ln["bbox"]:
                    break
                if nx["bbox"][1] - lns[-1]["bbox"][3] > pitch * CONFIG["para_gap_factor"]:
                    break
                lns.append(nx); j += 1
            text = join_wrapped([l["text"] for l in lns])
            if label == "example":
                b = mk(bidx, "example", lns, text=text)
            else:
                b = mk(bidx, "note", lns, label=label.capitalize(), text=text)
            if ln.get("icon_glyph"):
                noise.append({"type": "icon_glyph", "subtype": "label_prefix",
                              "text": ln["icon_glyph"],
                              "bbox": [round(v, 1) for v in ln["bbox"]],
                              "reason": "icon glyph OCR'd before '%s' label" % label})
                b["text"] = join_wrapped([l["text"].split(None, 1)[-1] if k == 0 else l["text"]
                                          for k, l in enumerate(lns)])
            blocks.append(b); bidx += 1
            i = j
            continue

        if role == "heading":
            b = mk(bidx, "heading", [ln], text=clean_text(ln["s"]),
                   level=ln["level"], heading_confidence=ln["heading_conf"])
            if ln["heading_conf"] == "low":
                b["flags"].append("non_bold_layout_candidate:not_promoted_to_section")
            blocks.append(b); bidx += 1
            i += 1
            continue

        if role in ("li_unordered", "li_ordered"):
            kind = "li_unordered"
            items, j = [], i
            cur = None
            while j < len(lines):
                nx = lines[j]
                if nx["role"] in ("li_unordered", "li_ordered"):
                    lvl = 2 if nx["marker"] == "o" else 1
                    cur = {"marker": nx["marker"] if nx["role"] == "li_unordered" else None,
                           "number": nx["marker"] if nx["role"] == "li_ordered" else None,
                           "level": lvl, "lines": [nx["item_text"]], "lns": [nx]}
                    items.append(cur)
                    j += 1
                elif nx["role"] == "paragraph" and cur is not None and nx["bbox"] \
                        and nx["aligned"] and nx["bbox"][0] >= items[-1]["lns"][-1]["bbox"][0] \
                        and nx["bbox"][1] - items[-1]["lns"][-1]["bbox"][3] <= pitch * 1.4:
                    cur["lines"].append(nx["text"]); cur["lns"].append(nx)
                    j += 1
                else:
                    break
            lns = [lnx for it in items for lnx in it["lns"]]
            b = mk(bidx, "list", lns,
                   ordered=items[0]["number"] is not None,
                   items=[{"level": it["level"],
                           "marker": it["marker"],
                           "number": it["number"],
                           "text": join_wrapped(it["lines"])} for it in items])
            blocks.append(b); bidx += 1
            i = j
            continue

        if role == "table_row":
            rows, j = [ln], i + 1
            while j < len(lines):
                nx = lines[j]
                if nx["role"] != "table_row" or not nx["bbox"]:
                    break
                if nx["bbox"][1] - rows[-1]["bbox"][3] > pitch * CONFIG["para_gap_factor"]:
                    break
                rows.append(nx); j += 1
            # column signature: median cell starts across rows
            starts = [c[0][1] for r in rows for c in r["cells"]]
            cols = []
            for st in sorted(starts):
                if not cols or st - cols[-1] > CONFIG["column_align_tol"]:
                    cols.append(st)
            def cell_of(r, col):
                for c in r["cells"]:
                    if abs(c[0][1] - col) <= CONFIG["column_align_tol"]:
                        return clean_text(" ".join(w[0] for w in c))
                return ""
            row_cells = [[clean_text(" ".join(w[0] for w in c)) for c in r["cells"]] for r in rows]
            # header detection: first row bolder (ink/stroke) than the rest
            headers = None
            if len(rows) >= 2:
                first_ink = statistics.mean(r["stats"]["ink"] for r in rows[:1])
                rest_ink = statistics.mean(r["stats"]["ink"] for r in rows[1:])
                first_stroke = statistics.mean(r["stats"]["med_stroke"] for r in rows[:1])
                rest_stroke = statistics.mean(r["stats"]["med_stroke"] for r in rows[1:])
                if rest_ink > 0 and (first_ink >= rest_ink * CONFIG["header_ink_factor"]
                                     or first_stroke >= rest_stroke * CONFIG["stroke_bold_ratio"]):
                    headers = row_cells[0]
                    body_rows = row_cells[1:]
                else:
                    body_rows = row_cells
            else:
                body_rows = row_cells
            b = mk(bidx, "table", rows,
                   columns=len(cols), column_starts=[round(c, 1) for c in cols],
                   headers=headers, rows=body_rows,
                   rows_ocr_order=row_cells)
            b["flags"].append("ocr_table_reconstruction:cell_split_by_x_gap")
            blocks.append(b); bidx += 1
            i = j
            continue

        if role == "fragment_cut":
            blocks.append(mk(bidx, "fragment", [ln], text=clean_text(ln["s"]),
                             fragment_reason="line cut off at page bottom"))
            blocks[-1]["flags"].append("cut_off_at_page_bottom")
            bidx += 1
            i += 1
            continue

        if role == "document_title":
            blocks.append(mk(bidx, "document_title", [ln], text=clean_text(ln["s"]),
                             level=1, title_kind="cover_area_title"))
            bidx += 1
            i += 1
            continue

        if role == "cover_meta" or (pno == 1 and RE_COVER_META.search(ln["s"])) \
                or (pno == 1 and RE_URLCONT.match(ln["s"]) and blocks
                    and blocks[-1]["type"] == "cover_meta"):
            if blocks and blocks[-1]["type"] == "cover_meta" \
                    and blocks[-1].get("bbox") and ln["bbox"] \
                    and ln["bbox"][1] - blocks[-1]["bbox"][3] <= pitch * 1.1:
                blocks[-1]["text"] = join_wrapped([blocks[-1]["text"], ln["s"]])
                blocks[-1]["bbox"] = [
                    min(blocks[-1]["bbox"][0], ln["bbox"][0]),
                    blocks[-1]["bbox"][1],
                    max(blocks[-1]["bbox"][2], ln["bbox"][2]),
                    max(blocks[-1]["bbox"][3], ln["bbox"][3])]
                i += 1
                continue
            blocks.append(mk(bidx, "cover_meta", [ln], text=clean_text(ln["s"])))
            bidx += 1
            i += 1
            continue

        # paragraph: gather contiguous paragraph lines
        lns, j = [ln], i + 1
        while j < len(lines):
            nx = lines[j]
            if not nx["s"] or nx["role"] != "paragraph" or not nx["bbox"] or not ln["bbox"]:
                break
            if nx["bbox"][1] - lns[-1]["bbox"][3] > pitch * CONFIG["para_gap_factor"]:
                break
            lns.append(nx); j += 1
        blocks.append(mk(bidx, "paragraph", lns, text=join_wrapped([l["text"] for l in lns])))
        bidx += 1
        i = j

    return blocks, noise


def page_cleaned_text(blocks: list) -> str:
    parts = []
    for b in blocks:
        if b["type"] == "list":
            for it in b["items"]:
                prefix = f"{it['number']} " if it["number"] else ("- " if it["marker"] else "")
                parts.append(prefix + it["text"])
        elif b["type"] == "table":
            for r in b.get("rows_ocr_order", b.get("rows", [])):
                parts.append(" | ".join(r))
        elif b.get("text"):
            parts.append(b["text"])
    return "\n\n".join(parts)


def process_page(task) -> dict:
    pdf_path, page, meta = task
    cls = classify_lines(page, pdf_path)
    blocks, noise = group_blocks(cls, page, meta)
    raw = page["text"]
    title = next((b["text"] for b in blocks if b["type"] == "heading"), None)
    cover_title = next((b["text"] for b in blocks if b["type"] == "document_title"), None)
    record = {
        "schema_version": SCHEMA_VERSION,
        "document_id": meta["document_id"],
        "page_number": page["page_number"],
        "source_page_hash": page["page_hash"],
        "is_cover_page": page["page_number"] == 1,
        "title": title,
        "cover_area_title": cover_title,
        "raw_text": raw,
        "cleaned_text": page_cleaned_text(blocks),
        "noise_spans": noise,
        "blocks": blocks,
        "detection": {
            "line_pitch_px": round(cls["pitch"], 1),
            "body_stroke_median": round(cls["body_median"], 4),
            "lines_reconstructed_aligned": all(ln.get("aligned", False) for ln in cls["lines"]),
            "issues": [],
        },
        "metadata": {
            "document_title": meta["title"],
            "source_pdf": meta["source_pdf"],
            "reference_pdf": meta["reference_pdf"],
            "category": meta["category"],
            "m2c_relevance": meta["m2c_relevance"],
            "source_url": meta["source_url"],
            "authority": 3,
            "authority_note": "page raw/cleaned text and blocks are OCR evidence of the "
                              "authoritative SAP Help Portal source document",
            "curation": {"authority": 2,
                         "source": "corpus_inventory.json",
                         "fields": ["document_title", "category", "m2c_relevance",
                                    "source_url", "reference_pdf"]},
            "ocr": {"engine": page["ocr_engine"], "dpi": page["ocr_settings"]["dpi"],
                    "mean_confidence": page["mean_confidence"],
                    "page_hash": page["page_hash"]},
        },
        "generated": {"authority": 1, "pipeline": f"canonicalize.py v{PIPELINE_VERSION}",
                      "processed_at": utcnow(), "config": CONFIG},
    }
    if noise:
        record["detection"]["issues"].append(
            f"{len(noise)} mechanical noise span(s) marked and excluded from cleaned_text")
    return record


# ------------------------------------------------------------- hierarchies --

def build_sections(doc_id: str, doc_title: str, pages: list) -> tuple:
    """Assign section ids/parents/paths across the document; tag page+blocks.

    Single pass over all blocks in reading order (page by page, block by block):
    high-confidence headings create sections; every other block belongs to the
    section that is currently open. Cross-page continuation is therefore
    structural, never inferred from adjacency of unrelated content.
    """
    sections = []
    root = {"section_id": f"{doc_id}_sec_000", "title": doc_title, "level": 1,
            "parent_section_id": None, "path": [doc_title],
            "page_start": None, "page_end": None, "block_ids": [],
            "derived_numbering": False, "confidence": "structural_root"}
    sections.append(root)
    last = {1: root}
    seq = 0
    issues = []
    current = root

    for page in pages:
        first_section_path = None
        for block in page["blocks"]:
            if block["type"] == "heading" and block.get("heading_confidence") == "high":
                level = block["level"]
                # topic-heading dedup: a top-level heading identical to the
                # curated document title describes the same node as the root
                if (level == 2
                        and block["text"].strip().lower() == doc_title.strip().lower()):
                    current = root
                    block["section_id"] = root["section_id"]
                    root["block_ids"].append(block["block_id"])
                    issues.append(f"page {page['page_number']}: top-level heading "
                                  f"'{block['text']}' equals document title; "
                                  f"merged into root section")
                    if first_section_path is None:
                        first_section_path = root["path"]
                    continue
                parent = next((last[lv] for lv in range(level - 1, 0, -1) if lv in last),
                              root)
                if parent is root and level > 2:
                    issues.append(f"page {page['page_number']}: level {level} heading "
                                  f"'{block['text']}' had no parent candidate; attached to root")
                seq += 1
                sec = {"section_id": f"{doc_id}_sec_{seq:03d}",
                       "title": block["text"], "level": level,
                       "parent_section_id": parent["section_id"],
                       "path": parent["path"] + [block["text"]],
                       "page_start": page["page_number"], "page_end": page["page_number"],
                       "block_ids": [block["block_id"]],
                       "derived_numbering": False,
                       "confidence": "high (bold/size detected)"}
                sections.append(sec)
                last[level] = sec
                for lv in range(level + 1, 5):
                    last.pop(lv, None)
                current = sec
            else:
                current = current if current is not None else root
                current["block_ids"].append(block["block_id"])
            block["section_id"] = current["section_id"]
            if first_section_path is None and block["type"] != "cover_meta":
                owner = next((s for s in sections if s["section_id"] == block["section_id"]),
                             root)
                first_section_path = owner["path"]
        page["section_path"] = first_section_path or root["path"]

    # section page ranges from owned blocks
    block_pages = {}
    for page in pages:
        for b in page["blocks"]:
            block_pages[b["block_id"]] = page["page_number"]
    for sec in sections:
        pgs = [block_pages[bid] for bid in sec["block_ids"] if bid in block_pages]
        if pgs:
            sec["page_start"], sec["page_end"] = min(pgs), max(pgs)
    # page -> sections present
    for page in pages:
        ids = []
        for b in page["blocks"]:
            if b["section_id"] not in ids:
                ids.append(b["section_id"])
        page["sections_on_page"] = [
            {"section_id": sid,
             "role": "starts_here" if any(
                 s["section_id"] == sid and s["page_start"] == page["page_number"]
                 for s in sections) else "continues"}
            for sid in ids]
    return sections, issues


# ------------------------------------------------------------- document IO --

def load_docs():
    inv = json.loads(INVENTORY_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_id = {d["document_id"]: d for d in manifest["documents"]}
    docs = []
    for num in sorted(inv["mapping"], key=int):
        meta = inv["mapping"][num]
        doc_id = f"doc_{int(num):03d}"
        m = by_id[doc_id]
        docs.append({"doc_id": doc_id,
                     "pdf_path": str(ROOT / "data" / meta["data_file"]),
                     "n_pages": m["num_pages"],
                     "meta": {
                         "document_id": doc_id,
                         "title": meta["title"],
                         "source_pdf": meta["data_file"],
                         "reference_pdf": meta["ref_file"],
                         "category": meta["category"],
                         "m2c_relevance": meta.get("m2c_relevance", ""),
                         "source_url": meta.get("url", ""),
                         "file_hash": m["file_hash"],
                     }})
    return docs


def assemble_document(doc, pages):
    doc_id = doc["doc_id"]
    sections, issues = build_sections(doc_id, doc["meta"]["title"], pages)

    # table continuation hints across page boundaries (detected, not merged)
    for a, b in zip(pages, pages[1:]):
        ta = next((bl for bl in reversed(a["blocks"]) if bl["type"] == "table"), None)
        tb = next((bl for bl in b["blocks"] if bl["type"] == "table"), None)
        if ta and tb and not any(bl["type"] == "heading" for bl in b["blocks"][:1]):
            if ta.get("column_starts") and tb.get("column_starts") and \
                    len(ta["column_starts"]) == len(tb["column_starts"]) and \
                    all(abs(x - y) <= CONFIG["column_align_tol"]
                        for x, y in zip(ta["column_starts"], tb["column_starts"])):
                ta.setdefault("flags", []).append(f"continued_on_next_page:p{b['page_number']}")
                tb.setdefault("flags", []).append(f"continues_from_previous_page:p{a['page_number']}")
                issues.append(f"table on p{a['page_number']}->p{b['page_number']} "
                              f"flagged as continuation (not merged; evidence kept per page)")

    stats = {"pages": len(pages), "blocks": 0, "headings": 0,
             "headings_by_level": {2: 0, 3: 0, 4: 0}, "heading_candidates_low": 0,
             "paragraphs": 0, "lists": 0, "list_items": 0, "tables": 0,
             "table_rows": 0, "notes": 0, "examples": 0, "fragments": 0,
             "cover_meta": 0, "noise_spans": 0, "characters_raw": 0,
             "characters_cleaned": 0}
    for p in pages:
        stats["noise_spans"] += len(p["noise_spans"])
        stats["characters_raw"] += len(p["raw_text"])
        stats["characters_cleaned"] += len(p["cleaned_text"])
        for b in p["blocks"]:
            stats["blocks"] += 1
            t = b["type"]
            if t == "heading":
                stats["headings"] += 1
                if b.get("heading_confidence") == "high":
                    stats["headings_by_level"][b["level"]] += 1
                else:
                    stats["heading_candidates_low"] += 1
            elif t == "paragraph":
                stats["paragraphs"] += 1
            elif t == "list":
                stats["lists"] += 1
                stats["list_items"] += len(b["items"])
            elif t == "table":
                stats["tables"] += 1
                stats["table_rows"] += len(b["rows"])
            elif t == "note":
                stats["notes"] += 1
            elif t == "example":
                stats["examples"] += 1
            elif t == "fragment":
                stats["fragments"] += 1
            elif t == "cover_meta":
                stats["cover_meta"] += 1

    doc_record = {
        "schema_version": SCHEMA_VERSION,
        "document_id": doc_id,
        "record_type": "document",
        "title": doc["meta"]["title"],
        "metadata": {
            **doc["meta"],
            "reference_pdf": doc["meta"]["reference_pdf"],
            "authority": 3,
            "curation": {"authority": 2, "source": "corpus_inventory.json",
                         "fields": ["title", "category", "m2c_relevance",
                                    "source_url", "reference_pdf"]},
            "source_pdf_file_hash": doc["meta"].get("file_hash"),
        },
        "cover_area_title": next((p["cover_area_title"] for p in pages
                                  if p.get("cover_area_title")), None),
        "page_count": len(pages),
        "pages": [{"page_number": p["page_number"], "title": p["title"],
                   "section_path": p["section_path"],
                   "sections_on_page": p["sections_on_page"],
                   "block_count": len(p["blocks"]),
                   "page_json": f"{doc_id}/page_{p['page_number']:03d}.json"}
                  for p in pages],
        "sections": sections,
        "stats": stats,
        "detection_issues": issues,
        "generated": {"authority": 1, "pipeline": f"canonicalize.py v{PIPELINE_VERSION}",
                      "processed_at": utcnow()},
    }
    return doc_record, sections, issues


def write_atomic(path: Path, data: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------- run --

def process_doc(doc, workers, force):
    doc_id, meta = doc["doc_id"], doc["meta"]
    doc_dir = PAGES_DIR / doc_id
    records, tasks, processed = {}, [], 0
    for pno in range(1, doc["n_pages"] + 1):
        ocr_path = OCR_DIR / doc_id / f"page_{pno:03d}.json"
        out_path = doc_dir / f"page_{pno:03d}.json"
        try:
            ocr = json.loads(ocr_path.read_text())
        except Exception as e:
            raise RuntimeError(f"{ocr_path} unreadable: {e}")
        if not force and out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
                if (prev.get("source_page_hash") == ocr["page_hash"]
                        and prev.get("schema_version") == SCHEMA_VERSION
                        and prev.get("generated", {}).get("pipeline") ==
                        f"canonicalize.py v{PIPELINE_VERSION}"
                        and prev.get("complete")
                        and "section_path" in prev):
                    records[pno] = prev
                    continue
            except Exception:
                pass
        tasks.append((doc["pdf_path"], ocr, meta, pno))

    print(f"[{doc_id}] {doc['n_pages']} pages: {len(records)} up-to-date, "
          f"{len(tasks)} to process", flush=True)

    if tasks:
        with Pool(processes=workers) as pool:
            for pno, record in pool.imap_unordered(process_page_task, tasks, chunksize=4):
                records[pno] = record
                processed += 1

    # assemble in memory (adds section ids/paths/roles + continuation hints)
    pages = [records[p] for p in sorted(records)]
    for p in pages:
        p["complete"] = True
    doc_record, sections, issues = assemble_document(doc, pages)

    # write only pages whose content actually changed (idempotent reruns)
    for pno in sorted(records):
        out_path = doc_dir / f"page_{pno:03d}.json"
        new = json.dumps(records[pno], indent=1)
        if not out_path.exists() or out_path.read_text() != new:
            write_atomic(out_path, new)
    return doc_record, processed, len(records) - processed


def process_page_task(t):
    pdf_path, ocr, meta, pno = t
    return pno, process_page((pdf_path, ocr, meta))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--docs", type=str, default="", help="comma separated doc ids")
    args = ap.parse_args()

    docs = load_docs()
    if args.docs:
        want = {d.strip() for d in args.docs.split(",")}
        docs = [d for d in docs if d["doc_id"] in want]
    t0 = time.time()
    CANON_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = total_skipped = 0
    doc_records = []
    for doc in docs:
        rec, processed, skipped = process_doc(doc, args.workers, args.force)
        doc_records.append(rec)
        total_processed += processed
        total_skipped += skipped
        s = rec["stats"]
        print(f"[{doc['doc_id']}] sections={len(rec['sections'])} "
              f"headings={s['headings']} (L2={s['headings_by_level'][2]} "
              f"L3={s['headings_by_level'][3]} L4={s['headings_by_level'][4]} "
              f"low={s['heading_candidates_low']}) tables={s['tables']} "
              f"lists={s['lists']} notes={s['notes']}", flush=True)

    with open(DOC_JSONL, "w") as f:
        for rec in doc_records:
            f.write(json.dumps(rec) + "\n")

    agg = {k: sum(d["stats"][k] for d in doc_records) for k in
           ["pages", "blocks", "headings", "paragraphs", "lists", "list_items",
            "tables", "table_rows", "notes", "examples", "fragments", "cover_meta",
            "noise_spans", "characters_raw", "characters_cleaned"]}
    agg["headings_by_level"] = {2: sum(d["stats"]["headings_by_level"][2] for d in doc_records),
                                3: sum(d["stats"]["headings_by_level"][3] for d in doc_records),
                                4: sum(d["stats"]["headings_by_level"][4] for d in doc_records)}
    agg["heading_candidates_low"] = sum(d["stats"]["heading_candidates_low"] for d in doc_records)
    manifest = {
        "generated_at": utcnow(),
        "pipeline": f"canonicalize.py v{PIPELINE_VERSION}",
        "schema_version": SCHEMA_VERSION,
        "processing": {"documents": len(doc_records),
                       "pages_processed": total_processed,
                       "pages_skipped_up_to_date": total_skipped,
                       "elapsed_seconds": round(time.time() - t0, 1)},
        "inputs": {"ocr": str(OCR_DIR.relative_to(ROOT)),
                   "inventory": "corpus_inventory.json",
                   "ocr_artifacts_modified": False},
        "totals": agg,
        "detection_settings": CONFIG,
        "authority_model": {"3": "SAP Help source content (text/blocks)",
                            "2": "curated reference metadata (corpus_inventory.json)",
                            "1": "generated/internal metadata (ids, detection, thresholds)"},
        "known_limitations": [
            "OCR substitutions (e.g. 'FPSO1', 'AP -MD') are intentionally NOT corrected",
            "non-bold layout-only heading candidates are flagged but do not create sections",
            "tables are reconstructed from x-gap clustering; complex/nested layouts may be imperfect",
            "cross-page tables are flagged as continuations but not merged (per-page evidence kept)",
        ],
    }
    write_atomic(CANON_MANIFEST, json.dumps(manifest, indent=1))
    print(f"\ncanonicalization done: {len(doc_records)} docs, "
          f"pages processed={total_processed} skipped={total_skipped}, "
          f"elapsed={manifest['processing']['elapsed_seconds']}s", flush=True)
    print("CANON_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
