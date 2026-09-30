#!/usr/bin/env python3
"""
VectorRAG - Part 4: Structure-Aware Parent/Child Chunking
=========================================================
Produces a retrieval-ready hierarchical chunk representation from the validated
canonical layer (data/processed/canonical). NO semantic rewriting: chunk
boundaries follow document structure first (sections -> blocks -> sentence/
item/row boundaries), token size second.

Levels:
  parent  = evidence/context container for one canonical section (never
            summarized; holds the section's direct content plus pointers to
            child sections and chunks). Canonical sections partition blocks,
            so direct content + child pointers is the complete section
            evidence without duplication.
  child   = retrieval chunk (will be embedded later); belongs to exactly one
            parent section; carries a metadata-only context prefix and keeps
            the exact cleaned source text separately
            (source_text vs retrieval_text)

Chunking algorithm (structure first, tokens second):
  1. If a whole section subtree fits the hard maximum, it is ONE chunk
     (small leaf sections therefore stay intact).
  2. Otherwise the section's direct blocks are packed to the target range and
     each child section is chunked recursively.
  3. Consecutive small sibling-subtree chunks are merged under their common
     parent (related sections, never unrelated ones); the sections a chunk
     covers are recorded explicitly (covers_section_ids + source_block_ids).
  4. An oversized single unit (huge paragraph/list/table) is split at
     sentence/item/row boundaries; table chunks always repeat the headers;
     prose splits carry a 1-sentence overlap.

Token counting: real Qwen3 BPE tokenizer, fully offline
(qwen-tokenizer 0.3.0 bundles the Qwen3 vocab; used through tiktoken).
The vocab is the Qwen3 family tokenizer used by Qwen3-Embedding models.

Outputs (idempotent: same canonical input -> byte-identical output):
    data/processed/chunks/parents.jsonl
    data/processed/chunks/children.jsonl
    data/processed/chunks/chunk_manifest.json
Nothing upstream is modified.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from qwen_tokenizer.tokenizer import get_tokenizer

ROOT = Path(__file__).resolve().parent
CANON = ROOT / "data" / "processed" / "canonical"
DOCS_JSONL = CANON / "documents.jsonl"
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"

PIPELINE_VERSION = "1.1"
SCHEMA_VERSION = "1.0"

CONFIG = {
    "target_min_tokens": 450,
    "target_max_tokens": 650,
    "hard_max_tokens": 800,
    "sibling_merge_min_tokens": 300,   # chunks smaller than this merge with
                                       # the next sibling-subtree chunk
    "absorb_threshold_tokens": 450,    # keep absorbing blocks while below this
    "oversize_unit_split_tokens": 700, # packing size for oversized-unit fragments
    "overlap_sentences": 1,            # prose overlap when splitting long units
    "table_rows_per_chunk_max": 30,
    "tokenizer": "Qwen3 BPE via qwen-tokenizer 0.3.0 (vocab qwen3_6.tiktoken, offline)",
}

RE_SENT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9\u201c(\"'])")

_TOK = None


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tok(text: str) -> int:
    global _TOK
    if _TOK is None:
        _TOK = get_tokenizer("Qwen/Qwen3-6B")
    return len(_TOK.encode(text))


# ------------------------------------------------------- block rendering --

def render_table(block: dict, rows=None, headers="keep") -> str:
    if headers == "keep":
        headers = block.get("headers")
    rows = block.get("rows_ocr_order") if rows is None else rows
    parts = []
    if headers:
        parts.append("Headers: " + " | ".join(headers))
    else:
        parts.append("Headers: (not detected in source)")
    parts.append("Rows:")
    for r in rows:
        parts.append(" | ".join(r))
    return "\n".join(parts)


def render_list(block: dict, items=None) -> str:
    items = block["items"] if items is None else items
    lines = []
    for it in items:
        indent = "  " * (it.get("level", 1) - 1)
        if it.get("number"):
            prefix = f"{it['number']} "
        elif it.get("marker") and it["marker"] not in ("e", "o"):
            prefix = f"{it['marker']} "
        else:
            prefix = "- "
        lines.append(indent + prefix + it["text"])
    return "\n".join(lines)


def unit_text(block: dict) -> str:
    t = block["type"]
    if t in ("paragraph", "fragment"):
        return block.get("text", "")
    if t == "note":
        label = block.get("label", "Note")
        text = block.get("text", "")
        if text.lower().startswith(label.lower()):
            return text  # canonical text already begins with the label word
        return f"{label}: {text}"
    if t == "example":
        text = block.get("text", "")
        if text.lower().startswith("example"):
            return text
        return f"Example: {text}"
    if t == "list":
        return render_list(block)
    if t == "table":
        return render_table(block)
    if t == "heading":
        return block.get("text", "")
    return block.get("text", "")


def is_child_worthy(block: dict) -> bool:
    """Cut-off/garbled fragments and cover/title blocks stay in the parent
    provenance, never become standalone retrieval content."""
    t = block["type"]
    if t in ("fragment", "document_title", "cover_meta"):
        return False
    return True


def is_substantive(block: dict) -> bool:
    """Child-worthy AND has real content (tiny figure remnants excluded)."""
    if not is_child_worthy(block):
        return False
    if block["type"] == "paragraph":
        words = len(block.get("text", "").split())
        if (words < 4 and len(block.get("text", "")) < 15
                and "in_graphic_region" in (block.get("flags") or [])):
            return False
    return True


# --------------------------------------------------------------- units --

class Unit:
    __slots__ = ("block", "text", "tokens", "kind")

    def __init__(self, block, text, kind):
        self.block = block
        self.text = text
        self.kind = kind
        self.tokens = tok(text)


def block_to_unit(block: dict) -> Unit | None:
    text = unit_text(block)
    if not text.strip():
        return None
    kind = block["type"]
    if kind == "list":
        kind = "procedure" if block.get("ordered") else "list"
    return Unit(block, text, kind)


def split_oversized(unit: Unit) -> list:
    """Split an oversized semantic unit at internal boundaries."""
    if unit.tokens <= CONFIG["hard_max_tokens"]:
        return [{"text": unit.text, "tokens": unit.tokens, "sub_kind": "whole",
                 "frag_total": 1, "extra": {}}]
    frags = []
    b = unit.block
    if unit.kind == "table":
        headers = b.get("headers")
        rows = b["rows_ocr_order"]
        groups, cur, cur_t = [], [], 0
        for r in rows:
            t = tok(" | ".join(r))
            if cur and (len(cur) >= CONFIG["table_rows_per_chunk_max"]
                        or cur_t + t > CONFIG["oversize_unit_split_tokens"]):
                groups.append(cur)
                cur, cur_t = [], 0
            cur.append(r)
            cur_t += t
        if cur:
            groups.append(cur)
        total = len(groups)
        for gi, g in enumerate(groups):
            text = render_table(b, rows=g, headers=headers)
            frags.append({"text": text, "tokens": tok(text), "sub_kind": "table_rows",
                          "frag_total": total,
                          "extra": {"row_start": sum(len(x) for x in groups[:gi]) + 1,
                                    "row_end": sum(len(x) for x in groups[:gi + 1]),
                                    "headers_repeated": True}})
        return frags
    if unit.kind in ("list", "procedure"):
        items = b["items"]
        groups, cur, cur_t = [], [], 0
        for it in items:
            t = tok(it["text"])
            if cur and cur_t + t > CONFIG["oversize_unit_split_tokens"]:
                groups.append(cur)
                cur, cur_t = [], 0
            cur.append(it)
            cur_t += t
        if cur:
            groups.append(cur)
        total = len(groups)
        for gi, g in enumerate(groups):
            text = render_list(b, items=g)
            frags.append({"text": text, "tokens": tok(text), "sub_kind": unit.kind,
                          "frag_total": total,
                          "extra": {"first_number": g[0].get("number"),
                                    "last_number": g[-1].get("number"),
                                    "item_range": [g[0].get("number") or 1,
                                                   g[-1].get("number") or len(g)]}})
        return frags
    # prose (paragraph / note / example): sentence packing, 1-sentence overlap
    sentences = [s for s in RE_SENT.split(unit.text) if s.strip()]
    if len(sentences) <= 1:
        words = unit.text.split()
        step = 140
        total = max(1, (len(words) + step - 1) // step)
        for gi, i in enumerate(range(0, len(words), step)):
            text = " ".join(words[i:i + step])
            frags.append({"text": text, "tokens": tok(text),
                          "sub_kind": "word_window", "frag_total": total,
                          "extra": {"no_sentence_boundary": True}})
        return frags
    packs, cur, cur_t = [], [], 0
    for s in sentences:
        t = tok(s)
        if cur and cur_t + t > CONFIG["oversize_unit_split_tokens"]:
            packs.append(cur)
            cur, cur_t = [], 0
        cur.append(s)
        cur_t += t
    if cur:
        packs.append(cur)
    total = len(packs)
    for gi, pack in enumerate(packs):
        text = " ".join(pack)
        extra = {}
        if gi > 0 and CONFIG["overlap_sentences"] > 0:
            text = " ".join(packs[gi - 1][-CONFIG["overlap_sentences"]:]) + " " + text
            extra["overlap_from_previous_fragment"] = CONFIG["overlap_sentences"]
        frags.append({"text": text, "tokens": tok(text), "sub_kind": "prose",
                      "frag_total": total, "extra": extra})
    return frags


def pack_units(units: list) -> list:
    """Greedy semantic packing of block units within one boundary level."""
    groups, cur, cur_t = [], [], 0
    for u in units:
        if not cur:
            cur, cur_t = [u], u.tokens
            continue
        t = cur_t + u.tokens
        if t <= CONFIG["target_max_tokens"]:
            cur.append(u)
            cur_t = t
        elif t <= CONFIG["hard_max_tokens"] and cur_t < CONFIG["absorb_threshold_tokens"]:
            cur.append(u)
            cur_t = t
        else:
            groups.append(cur)
            cur, cur_t = [u], u.tokens
    if cur:
        groups.append(cur)
    return groups


# ------------------------------------------------------------ recursion --

class TreeChunker:
    def __init__(self, doc, sections, blocks_by_sec):
        self.doc = doc
        self.sections = {s["section_id"]: s for s in sections}
        self.children_of = {}
        for s in sections:
            self.children_of.setdefault(s["parent_section_id"], []).append(s["section_id"])
        for k in self.children_of:
            self.children_of[k].sort()
        self.blocks = blocks_by_sec
        self.btok = {}
        self.notes = []
        for sid, blocks in blocks_by_sec.items():
            for b in blocks:
                self.btok[b["block_id"]] = tok(unit_text(b)) if is_child_worthy(b) else 0

        # section tree token totals (child-worthy content only)
        self.subtree_tokens = {}
        for s in sections:
            if s["section_id"] not in self.subtree_tokens:
                self._compute_subtree(s["section_id"])
        self._direct_children_cache = {}

    def child_secs(self, sid):
        return self.children_of.get(sid, [])

    def _compute_subtree(self, sid):
        total = 0
        for b in self.blocks.get(sid, []):
            if is_child_worthy(b):
                total += self.btok[b["block_id"]]
        for c in self.child_secs(sid):
            total += self._compute_subtree(c)
        self.subtree_tokens[sid] = total
        return total

    def subtree_block_ids(self, sid):
        out = []
        for b in self.blocks.get(sid, []):
            if is_child_worthy(b):
                out.append(b["block_id"])
        for c in self.child_secs(sid):
            out += self.subtree_block_ids(c)
        return out

    def sec_blocks(self, sid, substantive_only=False, include_headings_of_self=False):
        out = []
        for b in self.blocks.get(sid, []):
            if b["type"] == "heading" and not include_headings_of_self:
                continue  # a section's own heading is represented by its metadata
            if substantive_only and not is_substantive(b):
                continue
            if not substantive_only and not is_child_worthy(b):
                continue
            out.append(b)
        return out

    def descendant_headings(self, sid):
        """Heading blocks of descendant sections (reading order)."""
        out = []
        for c in self.child_secs(sid):
            for b in self.blocks.get(c, []):
                if b["type"] == "heading":
                    out.append(b)
            out += self.descendant_headings(c)
        return out

    def subtree_content_blocks(self, sid):
        """All child-worthy blocks of the subtree rooted at sid, in document
        order; includes descendant section headings (so subtree or merged
        chunks keep their internal structure), excludes sid's own heading."""
        out = []
        for b in self.blocks.get(sid, []):
            if b["type"] != "heading" and is_child_worthy(b):
                out.append(b)
        for c in self.child_secs(sid):
            for b in self.blocks.get(c, []):
                if b["type"] == "heading":
                    out.append(b)
            out += self.subtree_content_blocks(c)
        return out

    def subtree_sections(self, sid):
        """sid plus all descendant section ids."""
        out = [sid]
        for c in self.child_secs(sid):
            out += self.subtree_sections(c)
        return out

    # ---------------------------------------------------------- building --

    def build(self, sid):
        """Return list of chunk records for the subtree rooted at sid."""
        total = self.subtree_tokens[sid]
        if total == 0:
            return []
        if total <= CONFIG["hard_max_tokens"] - 20:
            # whole subtree in one chunk (small sections stay intact); the
            # 20-token margin absorbs "\n\n" join tokens counted at chunk level
            blocks = self.subtree_content_blocks(sid)
            return [self.mk_chunk(sid, blocks, covers=self.subtree_sections(sid))] \
                if blocks else []

        chunks = []
        direct = self.sec_blocks(sid, substantive_only=True)
        units = [u for u in (block_to_unit(b) for b in direct) if u]
        for grp in pack_units(units):
            if len(grp) == 1 and grp[0].tokens > CONFIG["hard_max_tokens"]:
                # oversized semantic unit: split at sentence/item/row boundaries
                self.notes.append(f"oversized_{grp[0].kind}_split")
                for fr in split_oversized(grp[0]):
                    chunks.append(self.mk_fragment_chunk(
                        sid, grp[0], fr, covers=self.subtree_sections(sid)))
                continue
            blocks = [u.block for u in grp]
            chunks.append(self.mk_chunk(sid, blocks, covers=[sid]))

        # recurse into child sections
        child_chunks = []
        for c in self.child_secs(sid):
            child_chunks += self.build(c)

        # merge consecutive small sibling-subtree chunks under this section
        merged = []
        for cc in child_chunks:
            prev = merged[-1] if merged else None
            if (prev is not None
                    and prev["owner"] != sid
                    and self.sections[prev["owner"]]["parent_section_id"] == sid
                    and self.sections[cc["owner"]]["parent_section_id"] == sid
                    and (prev["token_count"] < CONFIG["sibling_merge_min_tokens"]
                         or cc["token_count"] < CONFIG["sibling_merge_min_tokens"])
                    and prev["token_count"] + cc["token_count"] <= CONFIG["target_max_tokens"]):
                merged[-1] = self.merge_chunks(prev, cc, sid)
            else:
                merged.append(cc)
        # boundary merge: absorb a small first child chunk into the parent's
        # last direct chunk (child content belongs inside the parent context)
        if chunks and merged:
            lastd, first = chunks[-1], merged[0]
            if (lastd["owner"] == sid
                    and first["token_count"] < CONFIG["sibling_merge_min_tokens"]
                    and lastd["token_count"] < CONFIG["absorb_threshold_tokens"]
                    and lastd["token_count"] + first["token_count"]
                    <= CONFIG["target_max_tokens"]):
                chunks[-1] = self.merge_chunks(lastd, first, sid)
                merged = merged[1:]
        chunks += merged
        return chunks

    def mk_fragment_chunk(self, owner_sid, unit, fr, covers):
        b = unit.block
        flags = sorted(set(b.get("flags") or [])
                       | {"oversized_unit_split",
                          f"fragment_{fr['frag_index'] + 1}_of_{fr['frag_total']}"})
        extra = fr.get("extra") or {}
        src_text = fr["text"]
        sec = self.sections[owner_sid]
        prefix = (f"Title: {self.doc['title']}\n"
                  f"Section: {' > '.join(sec['path'])}\n"
                  f"Document: {self.doc['document_id']}\n\nContent:\n")
        pages = [b["page_number"]]
        kind = ("procedure" if (b["type"] == "list" and b.get("ordered"))
                else b["type"])
        rec = {
            "owner": owner_sid,
            "_blocks": [b],
            "covers_section_ids": sorted(set(covers)),
            "source_block_ids": [b["block_id"]],
            "source_text": src_text,
            "retrieval_text": prefix + src_text,
            "token_count": fr["tokens"],
            "character_count": len(src_text),
            "start_page": min(pages),
            "end_page": max(pages),
            "source_pages": sorted(set(pages)),
            "quality_flags": flags,
            "chunk_types": [kind],
            "contains_note": b["type"] == "note",
            "contains_example": b["type"] == "example",
        }
        for k, v in extra.items():
            rec[f"fragment_{k}"] = v
        return rec

    def mk_chunk(self, owner_sid, blocks, covers):
        parts, pages, flags, kinds = [], [], set(), set()
        for b in blocks:
            parts.append(unit_text(b))
            pages.append(b["page_number"])
            flags.update(b.get("flags") or [])
            kinds.add("procedure" if (b["type"] == "list" and b.get("ordered"))
                      else b["type"])
        source_text = "\n\n".join(p for p in parts if p.strip())
        sec = self.sections[owner_sid]
        prefix = (f"Title: {self.doc['title']}\n"
                  f"Section: {' > '.join(sec['path'])}\n"
                  f"Document: {self.doc['document_id']}\n\nContent:\n")
        return {
            "owner": owner_sid,
            "_blocks": blocks,
            "covers_section_ids": sorted(set(covers)),
            "source_block_ids": [b["block_id"] for b in blocks],
            "source_text": source_text,
            "retrieval_text": prefix + source_text,
            "token_count": tok(source_text),
            "character_count": len(source_text),
            "start_page": min(pages),
            "end_page": max(pages),
            "source_pages": sorted(set(pages)),
            "quality_flags": sorted(flags),
            "chunk_types": sorted(kinds),
            "contains_note": any(b["type"] == "note" for b in blocks),
            "contains_example": any(b["type"] == "example" for b in blocks),
        }

    def merge_chunks(self, a, b, new_owner):
        covers = sorted(set(a["covers_section_ids"]) | set(b["covers_section_ids"]))
        blocks = a["_blocks"] + b["_blocks"]
        mc = self.mk_chunk(new_owner, blocks, covers)
        mc["quality_flags"] = sorted(set(mc["quality_flags"])
                                     | {"merged_sibling_sections"})
        return mc


def process_doc(rec, pages_json):
    doc_id = rec["document_id"]
    doc_meta = {
        "document_id": doc_id,
        "title": rec["metadata"]["title"],
        "source_pdf": rec["metadata"]["source_pdf"],
        "reference_pdf": rec["metadata"]["reference_pdf"],
        "category": rec["metadata"]["category"],
        "m2c_relevance": rec["metadata"]["m2c_relevance"],
        "source_url": rec["metadata"]["source_url"],
    }
    blocks_by_sec = {}
    for p in pages_json:
        for b in p["blocks"]:
            blocks_by_sec.setdefault(b["section_id"], []).append(b)

    tc = TreeChunker(doc_meta, rec["sections"], blocks_by_sec)
    root_id = f"{doc_id}_sec_000"
    raw_chunks = tc.build(root_id)

    # finalize children (strip helper fields, add ids/metadata)
    children = []
    per_parent = {}
    for c in raw_chunks:
        owner = c.pop("owner")
        blocks = c.pop("_blocks", None)
        sec = tc.sections[owner]
        idx = per_parent.get(owner, 0) + 1
        per_parent[owner] = idx
        c["chunk_id"] = f"{owner}_chunk_{idx:03d}"
        c["parent_id"] = owner
        c["chunk_index"] = idx
        c["document_id"] = doc_id
        c["section_id"] = owner
        c["section_title"] = sec["title"]
        c["section_path"] = sec["path"]
        c["parent_section_id"] = sec["parent_section_id"]
        c["schema_version"] = SCHEMA_VERSION
        c["record_type"] = "child"
        children.append(c)
    # keep deterministic order: by (parent_id, chunk_index)
    children.sort(key=lambda c: (c["parent_id"], c["chunk_index"]))

    # parents: one per canonical section; coverage-based chunk mapping
    children_of = tc.children_of
    subtree_secs = {s["section_id"]: set(tc.subtree_sections(s["section_id"]))
                    for s in rec["sections"]}
    parents = []
    for s in rec["sections"]:
        sid = s["section_id"]
        blocks = tc.sec_blocks(sid)  # direct, incl. fragments/cover (evidence)
        src_parts = [unit_text(b) for b in blocks if unit_text(b).strip()]
        source_text = "\n\n".join(src_parts)
        kids = [c for c in children
                if subtree_secs[sid] & set(c["covers_section_ids"])]
        retrievable = tc.subtree_tokens[sid] > 0 and bool(kids)
        flags = []
        if s.get("confidence", "").startswith("low"):
            flags.append("low_confidence_heading")
        if any(b["type"] == "fragment" for b in blocks):
            flags.append("contains_fragments")
        if not retrievable and tc.subtree_tokens[sid] == 0:
            flags.append("non_retrievable:no_direct_content")
        elif not retrievable:
            flags.append("non_retrievable:content_not_chunkable")
        pages_ = sorted({b["page_number"] for b in blocks})
        parents.append({
            "parent_id": sid,
            "document_id": doc_id,
            "record_type": "parent",
            "section_title": s["title"],
            "section_path": s["path"],
            "section_level": s["level"],
            "parent_section_id": s["parent_section_id"],
            "child_section_ids": children_of.get(sid, []),
            "child_chunk_ids": [c["chunk_id"] for c in kids],
            "page_start": min(pages_) if pages_ else None,
            "page_end": max(pages_) if pages_ else None,
            "source_pages": pages_,
            "block_ids": [b["block_id"] for b in blocks],
            "subtree_block_ids": tc.subtree_block_ids(sid),
            "retrievable": retrievable,
            "source_text": source_text,
            "token_count": tok(source_text) if source_text else 0,
            "character_count": len(source_text),
            "subtree_token_count": tc.subtree_tokens[sid],
            "quality_flags": sorted(flags),
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "title": doc_meta["title"],
                "source_pdf": doc_meta["source_pdf"],
                "reference_pdf": doc_meta["reference_pdf"],
                "category": doc_meta["category"],
                "m2c_relevance": doc_meta["m2c_relevance"],
                "source_url": doc_meta["source_url"],
                "authority": 3,
            },
        })
    return parents, children, tc.notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    doc_records = [json.loads(l) for l in DOCS_JSONL.read_text().splitlines() if l.strip()]
    if args.docs:
        want = set(args.docs.split(","))
        doc_records = [d for d in doc_records if d["document_id"] in want]

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_parents, all_children = [], []
    stats_notes = []
    for rec in doc_records:
        pages_json = [json.loads((CANON / rec["document_id"] /
                                  f"page_{p['page_number']:03d}.json").read_text())
                      for p in rec["pages"]]
        parents, children, notes = process_doc(rec, pages_json)
        all_parents += parents
        all_children += children
        stats_notes += [f"{rec['document_id']}:{n}" for n in notes]
        print(f"[{rec['document_id']}] parents={len(parents)} children={len(children)}",
              flush=True)

    all_parents.sort(key=lambda p: p["parent_id"])
    all_children.sort(key=lambda c: c["chunk_id"])
    (CHUNKS_DIR / "parents.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in all_parents))
    (CHUNKS_DIR / "children.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in all_children))

    toks = sorted(c["token_count"] for c in all_children)
    n = len(toks)
    in_range = sum(1 for t in toks
                   if CONFIG["target_min_tokens"] <= t <= CONFIG["target_max_tokens"])
    pages_repr = sorted({(c["document_id"], p) for c in all_children
                         for p in c["source_pages"]})
    flagged = sum(1 for c in all_children
                  if any(f not in ("low_confidence_words:0",) for f in c["quality_flags"]))
    manifest = {
        "generated_at": utcnow(),
        "pipeline": f"chunker.py v{PIPELINE_VERSION}",
        "schema_version": SCHEMA_VERSION,
        "tokenizer": CONFIG["tokenizer"],
        "token_counting": "real BPE token counts (offline Qwen3 vocab)",
        "settings": CONFIG,
        "strategy": (
            "structure-first: a section subtree that fits the hard max is one chunk; "
            "bigger subtrees are chunked at direct-block boundaries and recursed; "
            "consecutive small sibling-subtree chunks merge under their common parent "
            "(covers_section_ids records what a chunk spans); oversized single units "
            "split at sentence/item/row boundaries, tables always repeat headers, "
            "prose fragments carry 1-sentence overlap; fragments/cover blocks stay in "
            "parent provenance only"),
        "pipeline_notes": stats_notes,
        "elapsed_seconds": round(time.time() - t0, 1),
        "totals": {
            "documents": len({p["document_id"] for p in all_parents}),
            "parent_sections": len(all_parents),
            "retrievable_parents": sum(1 for p in all_parents if p["retrievable"]),
            "non_retrievable_parents": sum(1 for p in all_parents if not p["retrievable"]),
            "child_chunks": n,
            "child_tokens_sum": sum(toks),
            "child_tokens_avg": round(sum(toks) / n, 1) if n else 0,
            "child_tokens_median": toks[n // 2] if n else 0,
            "child_tokens_min": toks[0] if n else 0,
            "child_tokens_max": toks[-1] if n else 0,
            "child_pct_within_450_650": round(100 * in_range / n, 1) if n else 0,
            "table_chunks": sum(1 for c in all_children if "table" in c["chunk_types"]),
            "procedure_or_list_chunks": sum(1 for c in all_children
                                            if "procedure" in c["chunk_types"]
                                            or "list" in c["chunk_types"]),
            "chunks_with_notes": sum(1 for c in all_children if c["contains_note"]),
            "chunks_with_examples": sum(1 for c in all_children if c["contains_example"]),
            "chunks_with_quality_flags": flagged,
            "pages_represented_in_children": len(pages_repr),
            "merged_sibling_chunks": sum(1 for c in all_children
                                         if "merged_sibling_sections" in c["quality_flags"]),
        },
        "inputs": {"canonical": str(CANON.relative_to(ROOT)),
                   "upstream_modified": False},
    }
    (CHUNKS_DIR / "chunk_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest["totals"], indent=1), flush=True)
    print("CHUNK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
