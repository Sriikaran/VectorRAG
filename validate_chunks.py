#!/usr/bin/env python3
"""
VectorRAG - Part 4 validation
=============================
Validates the chunk layer against the canonical layer by reading ACTUAL FILES
from disk. Prints PASS/FAIL per check; exits non-zero on any failure.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANON = ROOT / "data" / "processed" / "canonical"
CHUNKS = ROOT / "data" / "processed" / "chunks"
ING = ROOT / "data" / "processed" / "manifest.json"

failures = []


def check(num, name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append((num, name, detail))


def main() -> int:
    ing = json.loads(ING.read_text())
    num_pages = {d["document_id"]: d["num_pages"] for d in ing["documents"]}

    parents = [json.loads(l) for l in (CHUNKS / "parents.jsonl").read_text().splitlines() if l.strip()]
    children = [json.loads(l) for l in (CHUNKS / "children.jsonl").read_text().splitlines() if l.strip()]
    parent_by_id = {p["parent_id"]: p for p in parents}

    # canonical ground truth
    canon_secs = {}      # section_id -> section record
    canon_blocks = {}    # block_id -> (document_id, section_id, page, type, ordered)
    doc_sections = {}    # doc -> [section ids]
    for line in (CANON / "documents.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        doc = rec["document_id"]
        doc_sections[doc] = []
        for s in rec["sections"]:
            canon_secs[s["section_id"]] = s
            doc_sections[doc].append(s["section_id"])
    for doc in doc_sections:
        for pinfo in json.loads((CANON / doc / "documents.jsonl").read_text())["pages"] \
                if False else []:
            pass
        for pf in sorted((CANON / doc).glob("page_*.json")):
            page = json.loads(pf.read_text())
            for b in page["blocks"]:
                canon_blocks[b["block_id"]] = (
                    doc, b["section_id"], b["page_number"], b["type"],
                    b.get("ordered", False))

    # 1 - all 29 documents represented
    docs_p = {p["document_id"] for p in parents}
    docs_c = {c["document_id"] for c in children}
    check(1, "all 29 documents represented (parents+children)",
          docs_p == set(num_pages) and docs_c == set(num_pages),
          f"parents cover {len(docs_p)}, children cover {len(docs_c)}")

    # 2 - all canonical sections represented unless explicitly non-retrievable.
    # A section counts as covered when a chunk covers the section itself OR any
    # of its descendants (pure container sections hold no direct content).
    secs_by_parent = {}
    for sid, s in canon_secs.items():
        secs_by_parent.setdefault(s["parent_section_id"], []).append(sid)

    def subtree_of(sid):
        out = [sid]
        for c in secs_by_parent.get(sid, []):
            out += subtree_of(c)
        return out

    missing, bad_nonretr, uncovered = [], [], []
    for doc, sids in doc_sections.items():
        doc_children = [c for c in children if c["document_id"] == doc]
        for sid in sids:
            p = parent_by_id.get(sid)
            if p is None:
                missing.append(sid)
                continue
            if p["retrievable"]:
                tree = set(subtree_of(sid))
                covered = any(tree & set(c["covers_section_ids"])
                              or c["parent_id"] == sid for c in doc_children)
                if not covered:
                    uncovered.append(sid)
            else:
                if not any(f.startswith("non_retrievable:") for f in p["quality_flags"]):
                    bad_nonretr.append(sid)
    check(2, "all canonical sections represented unless marked non-retrievable",
          not missing and not bad_nonretr and not uncovered,
          f"missing={len(missing)} unflagged-nonretr={len(bad_nonretr)} "
          f"uncovered-retrievable={len(uncovered)}")

    # 3 - every child has exactly one parent (single-valued, resolvable, owner match)
    bad_parent = [c["chunk_id"] for c in children
                  if not c.get("parent_id") or c["parent_id"] not in parent_by_id
                  or c["section_id"] != c["parent_id"]]
    check(3, "every child chunk has exactly one parent (owner = parent section)",
          not bad_parent, f"{len(bad_parent)} bad")

    # 4 - every parent exists (child parent refs + parent parent refs inside doc)
    orphan_children = [c["chunk_id"] for c in children if c["parent_id"] not in parent_by_id]
    orphan_parents = [p["parent_id"] for p in parents
                      if p["parent_section_id"] is not None
                      and p["parent_section_id"] not in parent_by_id]
    check(4, "every parent/child reference resolves to an existing parent",
          not orphan_children and not orphan_parents,
          f"orphan children={len(orphan_children)}, orphan parents={len(orphan_parents)}")

    # 5/6 - duplicate ids
    dup_c = len(children) - len({c["chunk_id"] for c in children})
    dup_p = len(parents) - len({p["parent_id"] for p in parents})
    check(5, "no duplicate chunk IDs", dup_c == 0, f"{dup_c} duplicates")
    check(6, "no duplicate parent IDs", dup_p == 0, f"{dup_p} duplicates")

    # 7 - valid document ids
    bad_doc = [c["chunk_id"] for c in children if c["document_id"] not in num_pages]
    check(7, "every chunk has a valid document_id", not bad_doc, f"{len(bad_doc)} bad")

    # 8 - valid section_path (matches canonical section path)
    bad_path = []
    for c in children:
        sec = canon_secs.get(c["section_id"])
        if not sec or c["section_path"] != sec["path"] or not c["section_path"]:
            bad_path.append(c["chunk_id"])
    check(8, "every chunk has a valid section_path (== canonical)", not bad_path,
          f"{len(bad_path)} bad")

    # 9-11 - page references
    bad_pages = bad_range = 0
    for c in children:
        n = num_pages[c["document_id"]]
        sp = c["source_pages"]
        if (not sp or not all(isinstance(x, int) and 1 <= x <= n for x in sp)
                or c["start_page"] not in sp or c["end_page"] not in sp):
            bad_pages += 1
        if c["start_page"] > c["end_page"]:
            bad_range += 1
    check(9, "every chunk has valid page references", bad_pages == 0, f"{bad_pages} bad")
    check(10, "start_page <= end_page", bad_range == 0, f"{bad_range} bad")
    bad_sp = sum(1 for c in children
                 if sorted(set(c["source_pages"])) != c["source_pages"])
    check(11, "source_pages are valid, deduplicated page lists", bad_sp == 0,
          f"{bad_sp} bad")

    # 12 - source_text non-empty for retrievable chunks (children always;
    # container parents satisfy retrievability through their covering chunks)
    empty_children = [c["chunk_id"] for c in children if not c["source_text"].strip()]
    empty_parents = []
    for p in parents:
        if not p["retrievable"] or p["source_text"].strip():
            continue
        tree = set(subtree_of(p["parent_id"]))
        if not any(tree & set(c["covers_section_ids"])
                   and c["source_text"].strip() for c in children
                   if c["document_id"] == p["document_id"]):
            empty_parents.append(p["parent_id"])
    check(12, "source_text non-empty for all retrievable chunks",
          not empty_children and not empty_parents,
          f"empty children={len(empty_children)}, "
          f"retrievable parents without any content={len(empty_parents)}")

    # 13 - retrieval_text contains the context prefix and preserves source_text
    bad_prefix = 0
    for c in children:
        rt, st = c["retrieval_text"], c["source_text"]
        if not (rt.startswith("Title: ") and "\nSection: " in rt
                and f"\nDocument: {c['document_id']}\n\nContent:\n" in rt
                and rt.endswith(st)):
            bad_prefix += 1
    check(13, "retrieval_text contains context prefix and exact source_text",
          bad_prefix == 0, f"{bad_prefix} bad")

    # 14 - token counts present (real tokenizer, positive)
    bad_tok = [c["chunk_id"] for c in children
               if not isinstance(c.get("token_count"), int) or c["token_count"] <= 0
               or not isinstance(c.get("character_count"), int)]
    check(14, "token/character counts present on every chunk", not bad_tok,
          f"{len(bad_tok)} bad")

    # 15 - no child crosses document boundaries (blocks all from own document)
    cross_doc = 0
    for c in children:
        for bid in c["source_block_ids"]:
            info = canon_blocks.get(bid)
            if info is None or info[0] != c["document_id"]:
                cross_doc += 1
    check(15, "no child chunk crosses document boundaries", cross_doc == 0,
          f"{cross_doc} foreign block references")

    # 16 - no child crosses unrelated section boundaries: every source block's
    #      canonical section is inside the chunk's declared coverage, and all
    #      covered sections are within the parent's subtree
    cross_sec = 0
    for c in children:
        covers = set(c["covers_section_ids"])
        subtree = set(doc_sections[c["document_id"]])
        # collect subtree of parent from canonical parent links
        subtree = {sid for sid in doc_sections[c["document_id"]]
                   if sid == c["parent_id"]
                   or _is_descendant(canon_secs, sid, c["parent_id"])}
        if not covers <= subtree:
            cross_sec += 1
            continue
        for bid in c["source_block_ids"]:
            info = canon_blocks.get(bid)
            if info is None or info[1] not in covers:
                cross_sec += 1
                break
    check(16, "no child chunk crosses unrelated section boundaries", cross_sec == 0,
          f"{cross_sec} violations")

    # 17 - tables never split without repeated headers
    bad_table = 0
    table_chunks = 0
    for c in children:
        if "table" not in c["chunk_types"]:
            continue
        table_chunks += 1
        cont = any(f.startswith("fragment_row_start") and not f.endswith("_of_1")
                   for f in c.keys() if f.startswith("fragment_row_start"))
        is_continuation = c.get("fragment_row_start", 1) > 1
        if is_continuation and "Headers: " not in c["source_text"]:
            bad_table += 1
    check(17, f"table chunks repeat headers on every continuation ({table_chunks} table chunks)",
          bad_table == 0, f"{bad_table} violations")

    # 18 - ordered procedures preserve step order across split fragments
    frag_order = {}
    for c in children:
        for bid in c["source_block_ids"]:
            info = canon_blocks.get(bid)
            if info and info[3] == "table" and c.get("fragment_row_start") is not None:
                frag_order.setdefault((c["parent_id"], bid), []).append(
                    (c["chunk_index"], c["fragment_row_start"]))
            if info and info[4] and c.get("fragment_item_range") is not None:
                frag_order.setdefault((c["parent_id"], bid), []).append(
                    (c["chunk_index"], c["fragment_item_range"][0]))
    bad_order = sum(1 for seq in frag_order.values()
                    if [x[0] for x in seq] != sorted(x[0] for x in seq)
                    or [x[1] for x in seq] != sorted(x[1] for x in seq))
    check(18, "split procedures/tables preserve order across fragments",
          bad_order == 0, f"{bad_order} violations in {len(frag_order)} split sequences")

    # 19 - IDs deterministic (pattern-derived from parent + index)
    bad_ids = [c["chunk_id"] for c in children
               if c["chunk_id"] != f"{c['parent_id']}_chunk_{c['chunk_index']:03d}"]
    check(19, "IDs are deterministic (parent_id + index pattern)", not bad_ids,
          f"{len(bad_ids)} bad")

    # 20 - rerunning chunking on unchanged canonical data changes nothing
    before_c = (CHUNKS / "children.jsonl").read_bytes()
    before_p = (CHUNKS / "parents.jsonl").read_bytes()
    r = subprocess.run([sys.executable, str(ROOT / "chunker.py")],
                       capture_output=True, text=True, timeout=900)
    after_c = (CHUNKS / "children.jsonl").read_bytes()
    after_p = (CHUNKS / "parents.jsonl").read_bytes()
    check(20, "re-running chunking reproduces identical chunk IDs and content",
          r.returncode == 0 and before_c == after_c and before_p == after_p,
          f"children identical={before_c == after_c}, parents identical={before_p == after_p}")

    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print("  -", f)
        return 1
    print("VALIDATION PASSED: all checks green - chunk layer verified on disk")
    return 0


def _is_descendant(canon_secs, sid, ancestor_id):
    cur = canon_secs.get(sid)
    guard = 0
    while cur is not None and guard < 10000:
        pid = cur.get("parent_section_id")
        if pid == ancestor_id:
            return True
        cur = canon_secs.get(pid)
        guard += 1
    return False


if __name__ == "__main__":
    sys.exit(main())
