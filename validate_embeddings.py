#!/usr/bin/env python3
"""
VectorRAG - Part 5 validation
=============================
Validates the embedding layer against children.jsonl by reading ACTUAL FILES
from disk. Prints PASS/FAIL per check; exits non-zero on any failure
(including when artifacts do not exist yet — that is an honest failure).

Checks:
 1. Exactly 453 expected chunk IDs
 2. Exactly one embedding per chunk
 3. No duplicate chunk IDs
 4. No missing chunk IDs
 5. All vectors have dimension 2560
 6. All values are finite
 7. All vectors are normalized (unit L2 within tolerance)
 8. Every embedding references an existing chunk_id
 9. Stored source/retrieval hashes match the current chunk content
10. Model identifier is correct (Qwen/Qwen3-Embedding-4B)
11. Model revision/configuration recorded
12. Rerunning the pipeline produces no unnecessary recomputation (plan-only
    reports cache_hits == expected)
13. Changing a single chunk causes only that chunk to be recomputed
    (executed on TEMP COPIES via EMBEDDING_INPUT_DIR/EMBEDDING_OUTPUT_DIR;
    real artifacts are never modified)
"""

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
EXPECTED_COUNT = 453
EXPECTED_DIM = 2560
EXPECTED_MODEL = "Qwen/Qwen3-Embedding-4B"
NORM_TOL = 1e-3

failures = []


def check(num, name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {num}. {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append((num, name, detail))


def main() -> int:
    emb_path = EMB_DIR / "embeddings.jsonl"
    manifest_path = EMB_DIR / "embedding_manifest.json"

    if not emb_path.exists():
        print(f"BLOCKED: {emb_path} does not exist - no embeddings were "
              f"generated in this environment (see models/README.md for the "
              f"recorded blockers). No fabricated validation.")
        return 1

    records = [json.loads(l) for l in emb_path.read_text().splitlines() if l.strip()]
    children = [json.loads(l) for l in (CHUNKS / "children.jsonl").read_text().splitlines()
                if l.strip()]
    child_by_id = {c["chunk_id"]: c for c in children}

    ids = [r["chunk_id"] for r in records]
    # 1 - expected count
    check(1, f"exactly {EXPECTED_COUNT} expected chunk IDs embedded",
          len(children) == EXPECTED_COUNT and len(records) == EXPECTED_COUNT,
          f"children={len(children)}, embeddings={len(records)}")
    # 2 - one embedding per chunk
    check(2, "exactly one embedding per chunk",
          len(ids) == len(set(ids)) == len(child_by_id),
          f"records={len(ids)}, unique={len(set(ids))}")
    # 3 - no duplicates
    dup = len(ids) - len(set(ids))
    check(3, "no duplicate chunk IDs", dup == 0, f"{dup} duplicates")
    # 4 - no missing ids
    missing = sorted(set(child_by_id) - set(ids))
    extra = sorted(set(ids) - set(child_by_id))
    check(4, "no missing chunk IDs", not missing and not extra,
          f"missing={len(missing)} extra={len(extra)}")
    # 5 - dimension
    bad_dim = [r["chunk_id"] for r in records if len(r["vector"]) != EXPECTED_DIM
               or r.get("dimension") != EXPECTED_DIM]
    check(5, f"all vectors have dimension {EXPECTED_DIM}", not bad_dim,
          f"{len(bad_dim)} bad")
    # 6 - finite
    bad_finite = [r["chunk_id"] for r in records
                  if not all(math.isfinite(x) for x in r["vector"])]
    check(6, "all vector values finite (no NaN/Inf)", not bad_finite,
          f"{len(bad_finite)} bad")
    # 7 - normalized
    bad_norm = []
    for r in records:
        n = math.sqrt(sum(x * x for x in r["vector"]))
        if abs(n - 1.0) > NORM_TOL:
            bad_norm.append((r["chunk_id"], round(n, 6)))
    check(7, f"all vectors normalized (|L2-1| <= {NORM_TOL})", not bad_norm,
          f"{len(bad_norm)} bad" + (f", e.g. {bad_norm[:3]}" if bad_norm else ""))
    # 8 - valid chunk references
    bad_ref = [r["chunk_id"] for r in records
               if r["chunk_id"] not in child_by_id
               or r["parent_id"] != child_by_id[r["chunk_id"]]["parent_id"]
               or r["document_id"] != child_by_id[r["chunk_id"]]["document_id"]
               or r["section_id"] != child_by_id[r["chunk_id"]]["section_id"]]
    check(8, "every embedding references an existing chunk (id/parent/doc/section)",
          not bad_ref, f"{len(bad_ref)} bad")
    # 9 - hashes match current content
    import hashlib
    def h(s):
        return hashlib.sha256(s.encode()).hexdigest()
    bad_hash = [r["chunk_id"] for r in records
                if r["chunk_id"] in child_by_id
                and (r["retrieval_text_hash"]
                     != h(child_by_id[r["chunk_id"]]["retrieval_text"])
                     or r["source_text_hash"]
                     != h(child_by_id[r["chunk_id"]]["source_text"]))]
    check(9, "stored source/retrieval hashes match current chunk content",
          not bad_hash, f"{len(bad_hash)} stale")
    # 10 - model identifier
    bad_model = [r["chunk_id"] for r in records if r.get("model") != EXPECTED_MODEL]
    check(10, f"model identifier is {EXPECTED_MODEL}", not bad_model,
          f"{len(bad_model)} bad")
    # 11 - revision/config recorded
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ok11 = all(r.get("model_revision") for r in records) and \
        bool(man.get("model_revision")) and bool(man.get("embedding_config"))
    check(11, "model revision and embedding configuration recorded", ok11,
          f"revision={man.get('model_revision')}")
    # 12 - rerun produces no unnecessary recomputation
    r = subprocess.run([sys.executable, str(ROOT / "embed_chunks.py"), "--plan-only"],
                       capture_output=True, text=True, timeout=600)
    try:
        plan = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:
        plan = {}
    check(12, "rerunning the pipeline recomputes nothing (all cache hits)",
          r.returncode == 0 and plan.get("cache_hits") == EXPECTED_COUNT
          and plan.get("would_embed") == 0,
          f"plan: hits={plan.get('cache_hits')}, would_embed={plan.get('would_embed')}")
    # 13 - single-chunk change recomputes only that chunk (TEMP COPIES ONLY)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_in = Path(td) / "chunks"
            tmp_out = Path(td) / "embeddings"
            shutil.copytree(CHUNKS, tmp_in)
            shutil.copytree(EMB_DIR, tmp_out)
            kids = [json.loads(l) for l in
                    (tmp_in / "children.jsonl").read_text().splitlines() if l.strip()]
            kids[0]["retrieval_text"] = kids[0]["retrieval_text"] + " "
            (tmp_in / "children.jsonl").write_text(
                "".join(json.dumps(c, sort_keys=True) + "\n" for c in kids))
            env = dict(os.environ, EMBEDDING_INPUT_DIR=str(tmp_in),
                       EMBEDDING_OUTPUT_DIR=str(tmp_out))
            r2 = subprocess.run([sys.executable, str(ROOT / "embed_chunks.py"),
                                 "--plan-only"], capture_output=True, text=True,
                                timeout=600, env=env)
            plan2 = json.loads(r2.stdout[r2.stdout.index("{"):])
            ok13 = (r2.returncode == 0 and plan2.get("would_embed") == 1
                    and plan2.get("cache_hits") == EXPECTED_COUNT - 1)
            check(13, "changing one chunk recomputes exactly that chunk",
                  ok13, f"plan: hits={plan2.get('cache_hits')}, "
                        f"would_embed={plan2.get('would_embed')}")
    except Exception as e:
        check(13, "changing one chunk recomputes exactly that chunk", False,
              f"{type(e).__name__}: {e}")

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
