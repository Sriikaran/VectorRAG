#!/usr/bin/env python3
"""
VectorRAG - Part 5 validation (0.6B ONNX profile)
=================================================
Validates the embedding layer against children.jsonl by reading ACTUAL FILES
from disk. Prints PASS/FAIL per check; exits non-zero on any failure —
including when no embeddings exist yet (an honest BLOCKED, never fabricated).

Checks:
 1. 453 expected chunk IDs
 2. 453 unique embeddings
 3. No missing chunk IDs
 4. No duplicate chunk IDs
 5. Dimension = 1024
 6. No NaN/Inf
 7. Unit normalization (|L2-1| <= tolerance)
 8. Every vector references a valid chunk (id/parent/doc/section)
 9. Hashes match current chunk content
10. Correct model/backend recorded (Qwen3-Embedding-0.6B-ONNX / onnx)
11. Revision + configuration recorded
12. Real idempotent rerun (plan-only reports full content-match coverage)
13. Original canonical/chunk artifacts remain unchanged
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EMB_DIR = ROOT / "data" / "processed" / "embeddings"
CHUNKS = ROOT / "data" / "processed" / "chunks"
CANON_DOC = ROOT / "data" / "processed" / "canonical" / "documents.jsonl"
EXPECTED_COUNT = 453
EXPECTED_DIM = 1024
EXPECTED_MODEL = "Qwen3-Embedding-0.6B-ONNX"
EXPECTED_BACKEND = "onnx"
NORM_TOL = 1e-3

failures = []


def check(num, name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append((num, name, detail))


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> int:
    emb_path = EMB_DIR / "embeddings.jsonl"
    manifest_path = EMB_DIR / "embedding_manifest.json"

    if not emb_path.exists():
        print(f"BLOCKED: {emb_path} does not exist - no embeddings were "
              f"generated in this environment (model acquisition failed; see "
              f"models/README.md). No fabricated validation.")
        return 1

    records = [json.loads(l) for l in emb_path.read_text().splitlines() if l.strip()]
    children = [json.loads(l) for l in (CHUNKS / "children.jsonl").read_text().splitlines()
                if l.strip()]
    child_by_id = {c["chunk_id"]: c for c in children}

    ids = [r["chunk_id"] for r in records]
    check(1, f"{EXPECTED_COUNT} expected chunk IDs",
          len(children) == EXPECTED_COUNT and len(records) == EXPECTED_COUNT,
          f"children={len(children)}, embeddings={len(records)}")
    check(2, f"{len(set(ids))} unique embeddings", len(set(ids)) == len(records),
          f"unique={len(set(ids))}")
    missing = sorted(set(child_by_id) - set(ids))
    extra = sorted(set(ids) - set(child_by_id))
    check(3, "no missing chunk IDs", not missing, f"missing={len(missing)}")
    dup = len(ids) - len(set(ids))
    check(4, "no duplicate chunk IDs", dup == 0, f"{dup} duplicates")
    bad_dim = [r["chunk_id"] for r in records
               if len(r["vector"]) != EXPECTED_DIM or r.get("dimension") != EXPECTED_DIM]
    check(5, f"dimension = {EXPECTED_DIM}", not bad_dim, f"{len(bad_dim)} bad")
    bad_finite = [r["chunk_id"] for r in records
                  if not all(math.isfinite(x) for x in r["vector"])]
    check(6, "no NaN/Inf values", not bad_finite, f"{len(bad_finite)} bad")
    bad_norm = []
    for r in records:
        n = math.sqrt(sum(x * x for x in r["vector"]))
        if abs(n - 1.0) > NORM_TOL:
            bad_norm.append((r["chunk_id"], round(n, 6)))
    check(7, f"unit normalization (|L2-1| <= {NORM_TOL})", not bad_norm,
          f"{len(bad_norm)} bad" + (f", e.g. {bad_norm[:3]}" if bad_norm else ""))
    bad_ref = [r["chunk_id"] for r in records
               if r["chunk_id"] not in child_by_id
               or r["parent_id"] != child_by_id[r["chunk_id"]]["parent_id"]
               or r["document_id"] != child_by_id[r["chunk_id"]]["document_id"]
               or r["section_id"] != child_by_id[r["chunk_id"]]["section_id"]]
    check(8, "every vector references a valid chunk", not bad_ref,
          f"{len(bad_ref)} bad")
    bad_hash = [r["chunk_id"] for r in records
                if r["chunk_id"] in child_by_id
                and (r["retrieval_text_hash"]
                     != sha_text(child_by_id[r["chunk_id"]]["retrieval_text"])
                     or r["source_text_hash"]
                     != sha_text(child_by_id[r["chunk_id"]]["source_text"]))]
    check(9, "stored hashes match current chunk content", not bad_hash,
          f"{len(bad_hash)} stale")
    bad_model = [r["chunk_id"] for r in records
                 if r.get("model") != EXPECTED_MODEL or r.get("backend") != EXPECTED_BACKEND]
    check(10, f"correct model/backend recorded ({EXPECTED_MODEL}/{EXPECTED_BACKEND})",
          not bad_model, f"{len(bad_model)} bad")
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ok11 = all(r.get("model_revision") for r in records) and \
        bool(man.get("model_revision")) and bool(man.get("embedding_config")) \
        and man.get("embedding_dimension") == EXPECTED_DIM
    check(11, "revision + embedding configuration recorded", ok11,
          f"revision={man.get('model_revision')}, "
          f"device={man.get('device')}, quantization={man.get('quantization')}")
    # 12 - real idempotent rerun (content-match coverage; the pinned revision is
    # part of the real cache key, plan-only reports content-level matches)
    r = subprocess.run([sys.executable, str(ROOT / "embed_chunks.py"), "--plan-only"],
                       capture_output=True, text=True, timeout=600)
    try:
        plan = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:
        plan = {}
    check(12, "rerun recomputes nothing (full content-match coverage)",
          r.returncode == 0 and plan.get("content_match_hits") == EXPECTED_COUNT
          and plan.get("would_embed") == 0,
          f"plan: hits={plan.get('content_match_hits')}, "
          f"would_embed={plan.get('would_embed')}")
    # 13 - original canonical/chunk artifacts unchanged (manifest-recorded hashes)
    if man.get("artifact_hashes"):
        changed = []
        for rel, h in man["artifact_hashes"].items():
            f = ROOT / rel
            if not f.exists() or sha_text(f.read_text()) != h:
                changed.append(rel)
        check(13, "original canonical/chunk artifacts remain unchanged",
              not changed, f"changed: {changed or 'none'}")
    else:
        check(13, "original canonical/chunk artifacts remain unchanged", False,
              "manifest does not record artifact_hashes (stale manifest?)")

    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print("  -", f)
        return 1
    print("VALIDATION PASSED: all checks green - embedding layer verified on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
