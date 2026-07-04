#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg2-binary",
# ]
# ///
"""
reconciler.py — V2 declarative corpus reconciler (docs/V2-REFACTOR-PLAN.md).

The DB is treated as a pure function of the scraped corpus: every run computes

    desired  = chunk the corpus (content-addressed by source_hash)
    actual   = what the chunks table currently holds
    plan     = per-bill diff -> {replace, delete}
    execute  = embed only NEW hashes (existing embeddings are reused by hash),
               then one transaction per bill: DELETE bill rows, INSERT desired
    verify   = the table must now equal the desired manifest EXACTLY, and
               ops.check_invariants() must be all-green
    promote  = mark the ops.reconcile_runs row success + refresh data versions

Fail-loud rules (this replaces v1's warn-and-continue):
  * corpus missing/empty, or post-run manifest mismatch  -> abort non-zero
  * plan wants to delete more than --max-delete-frac of the bills in the DB
    -> abort (protects against a corrupt/empty scrape wiping retrieval)
  * any red invariant after execution -> run recorded failed, exit non-zero

Crash-safety: per-bill transactions mean a killed run leaves a valid subset
(invariants stay green); the next run supersedes abandoned 'running' audit
rows and converges the remaining diff.

Usage:
    python reconciler.py --congress 119                     # corpus walk
    python reconciler.py --desired-json fixtures.json ...   # drill seam
    python reconciler.py ... --backend fake                 # no OpenAI
    python reconciler.py ... --dry-run                      # plan only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import socket
import struct
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from upserter import chunk_source_hash, vector_literal

DEFAULT_DSN = os.environ.get("PG_CONNECTION_STRING", "")
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
DEFAULT_MAX_DELETE_FRAC = 0.2
EMBED_BATCH = 128

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("reconciler")


class ReconcileError(RuntimeError):
    """Fatal, deliberate abort — recorded on the audit row, exits non-zero."""


# ---------------------------------------------------------------------------
# Desired state
# ---------------------------------------------------------------------------

def desired_from_corpus(corpus_root: Path, congress: int) -> list[dict]:
    """Chunk the scraped corpus with the SAME chunker v1 uses."""
    from chunker import apply_chunk_filters, process_congress  # heavy import, lazy

    bills_dir = corpus_root / f"bills_{congress}"
    if not bills_dir.is_dir():
        raise ReconcileError(f"corpus not found: {bills_dir}")
    chunks, _stats = process_congress(bills_dir, max_tokens=512, overlap=50)
    chunks, _below, _capped = apply_chunk_filters(chunks, min_tokens=30, cap=200)
    if not chunks:
        raise ReconcileError(f"corpus produced zero chunks under {bills_dir}")
    return chunks


def desired_from_json(path: Path) -> list[dict]:
    """Drill seam: inject the desired chunk list directly (see plan Phase 3)."""
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ReconcileError(f"desired-json must be a non-empty list: {path}")
    return chunks


def index_desired(chunks: list[dict]) -> dict[str, dict[str, dict]]:
    """bill_uid -> source_hash -> chunk. Duplicate hashes are a corpus bug."""
    desired: dict[str, dict[str, dict]] = defaultdict(dict)
    for chunk in chunks:
        h = chunk_source_hash(chunk)
        uid = str(chunk.get("canonical_bill_id") or chunk.get("bill_id"))
        if not uid or uid == "None":
            raise ReconcileError(f"chunk without a bill identity: {chunk.get('chunk_index')}")
        if h in desired[uid]:
            raise ReconcileError(f"duplicate source_hash {h[:12]}… within {uid}")
        desired[uid][h] = chunk
    return dict(desired)


# ---------------------------------------------------------------------------
# Actual state + plan
# ---------------------------------------------------------------------------

def resolve_schema(cur) -> str:
    """Post-swap the table lives in nlp; pre-swap in nlp_stage."""
    for schema in ("nlp", "nlp_stage"):
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.chunks",))
        if cur.fetchone()[0] is not None:
            return schema
    raise ReconcileError("no v2 chunks table found (nlp.chunks / nlp_stage.chunks) — is migration 0012 applied?")


def fetch_actual(cur, schema: str, congress: int) -> dict[str, set[str]]:
    cur.execute(
        f"SELECT bill_uid, source_hash FROM {schema}.chunks WHERE congress = %s",
        (congress,))
    actual: dict[str, set[str]] = defaultdict(set)
    for bill_uid, source_hash in cur:
        actual[bill_uid].add(source_hash)
    return dict(actual)


def build_plan(desired: dict[str, dict[str, dict]], actual: dict[str, set[str]],
               max_delete_frac: float) -> dict:
    replace = [uid for uid, chunks in desired.items()
               if set(chunks.keys()) != actual.get(uid, set())]
    delete = [uid for uid in actual if uid not in desired]
    new_hashes = {h for uid in replace for h in desired[uid].keys()
                  if h not in actual.get(uid, set())}
    if actual:
        frac = len(delete) / len(actual)
        if delete and frac > max_delete_frac:
            raise ReconcileError(
                f"plan deletes {len(delete)}/{len(actual)} bills "
                f"({frac:.0%}) > --max-delete-frac {max_delete_frac:.0%} — "
                "refusing; a corrupt or partial scrape looks exactly like this")
    return {"replace": replace, "delete": delete, "new_hashes": new_hashes}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def fake_embedding(source_hash: str, dimensions: int) -> list[float]:
    """Deterministic pseudo-embedding derived from the hash (drills only)."""
    raw = b""
    seed = source_hash.encode()
    while len(raw) < dimensions * 4:
        seed = hashlib.sha256(seed).digest()
        raw += seed
    vals = [struct.unpack_from(">i", raw, i * 4)[0] / 2**31 for i in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def embed_new(chunks_by_hash: dict[str, dict], backend: str, model: str,
              dimensions: int) -> dict[str, list[float]]:
    if not chunks_by_hash:
        return {}
    if backend == "fake":
        return {h: fake_embedding(h, dimensions) for h in chunks_by_hash}
    from openai import OpenAI  # only needed on the real path
    client = OpenAI()
    out: dict[str, list[float]] = {}
    hashes = list(chunks_by_hash.keys())
    for i in range(0, len(hashes), EMBED_BATCH):
        batch = hashes[i:i + EMBED_BATCH]
        texts = [str(chunks_by_hash[h].get("text", "")) for h in batch]
        resp = client.embeddings.create(model=model, input=texts, dimensions=dimensions)
        for h, item in zip(batch, resp.data):
            out[h] = item.embedding
        log.info(f"embedded {min(i + EMBED_BATCH, len(hashes))}/{len(hashes)} new chunks")
    return out


def load_reusable_embeddings(cur, schema: str, bill_uids: list[str]) -> dict[str, tuple[str, str]]:
    """source_hash -> (embedding text, model) for bills about to be replaced."""
    if not bill_uids:
        return {}
    cur.execute(
        f"SELECT source_hash, embedding::text, embedding_model FROM {schema}.chunks"
        " WHERE bill_uid = ANY(%s)", (bill_uids,))
    return {h: (emb, model) for h, emb, model in cur}


# ---------------------------------------------------------------------------
# Execute + verify
# ---------------------------------------------------------------------------

def chunk_row(chunk: dict, h: str, embedding: str, model: str) -> tuple:
    return (
        h,
        str(chunk.get("canonical_bill_id") or chunk.get("bill_id")),
        int(chunk.get("congress")),
        str(chunk.get("type", "")),
        str(chunk.get("number", "")),
        "section",
        int(chunk.get("chunk_index", 0)),
        chunk.get("short_title"),
        chunk.get("status"),
        chunk.get("section_enum"),
        chunk.get("section_header"),
        str(chunk.get("text", "")),
        int(chunk.get("tokens", 0)),
        embedding,
        model,
    )


def execute_plan(conn, schema: str, plan: dict, desired: dict[str, dict[str, dict]],
                 new_embeddings: dict[str, list[float]],
                 reusable: dict[str, tuple[str, str]], model: str) -> dict:
    executed = {"add": 0, "update": 0, "delete": 0}
    crash_after = int(os.environ.get("RECONCILE_CRASH_AFTER_BILLS", "0"))
    done_bills = 0
    with conn.cursor() as cur:
        for uid in plan["delete"]:
            cur.execute(f"DELETE FROM {schema}.chunks WHERE bill_uid = %s", (uid,))
            executed["delete"] += cur.rowcount
            conn.commit()
        for uid in plan["replace"]:
            rows = []
            for h, chunk in desired[uid].items():
                if h in new_embeddings:
                    emb, emb_model = vector_literal(new_embeddings[h]), model
                    executed["add"] += 1
                else:
                    emb, emb_model = reusable[h]
                    executed["update"] += 1
                rows.append(chunk_row(chunk, h, emb, emb_model))
            cur.execute(f"DELETE FROM {schema}.chunks WHERE bill_uid = %s", (uid,))
            execute_values(cur, f"""
                INSERT INTO {schema}.chunks
                    (source_hash, bill_uid, congress, bill_type, bill_number,
                     chunk_type, chunk_index, title, status, section_enum,
                     section_header, body, token_count, embedding, embedding_model)
                VALUES %s""", rows)
            conn.commit()
            done_bills += 1
            if crash_after and done_bills >= crash_after:
                log.error(f"RECONCILE_CRASH_AFTER_BILLS={crash_after}: simulating crash")
                os._exit(1)  # drill seam: hard kill, no cleanup
    return executed


def verify(cur, schema: str, congress: int, desired: dict[str, dict[str, dict]]) -> int:
    """The table must equal the desired manifest exactly. Returns chunk count."""
    want = sorted(h for chunks in desired.values() for h in chunks)
    cur.execute(
        f"SELECT coalesce(md5(string_agg(source_hash, ',' ORDER BY source_hash)), 'empty'),"
        f" count(*) FROM {schema}.chunks WHERE congress = %s", (congress,))
    got_md5, got_count = cur.fetchone()
    want_md5 = hashlib.md5(",".join(want).encode()).hexdigest() if want else "empty"
    if got_md5 != want_md5 or got_count != len(want):
        raise ReconcileError(
            f"post-run manifest mismatch: DB has {got_count} chunks (md5 {got_md5[:12]}…), "
            f"desired {len(want)} (md5 {want_md5[:12]}…) — NOT promoting")
    cur.execute("SELECT name, detail FROM ops.check_invariants() WHERE NOT ok")
    red = cur.fetchall()
    if red:
        raise ReconcileError("red invariants after execute: "
                             + "; ".join(f"{n}: {d}" for n, d in red))
    return got_count


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_start(dsn: str, run_id: str, congress: int | None, model: str) -> None:
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        # Supersede abandoned runs: a crashed reconciler leaves 'running';
        # the next run is the recovery, so mark predecessors aborted.
        cur.execute("""
            UPDATE ops.reconcile_runs SET status = 'aborted', finished_at = now(),
                   error = coalesce(error, 'superseded by ' || %s)
             WHERE status = 'running'""", (run_id,))
        cur.execute("""
            INSERT INTO ops.reconcile_runs (run_id, congress, embedding_model, git_sha)
            VALUES (%s, %s, %s, %s)""",
            (run_id, congress, model, os.environ.get("GIT_SHA")))


def audit_finish(dsn: str, run_id: str, status: str, fields: dict) -> None:
    sets = ", ".join(f"{k} = %s" for k in fields)
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE ops.reconcile_runs SET status = %s, finished_at = now(), {sets}"
            " WHERE run_id = %s",
            [status, *fields.values(), run_id])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Declarative corpus reconciler (v2)")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--congress", type=int, default=None)
    p.add_argument("--corpus-root", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--desired-json", default=None,
                   help="drill seam: read desired chunks from JSON instead of the corpus")
    p.add_argument("--backend", choices=["openai", "fake"], default="openai")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    p.add_argument("--max-delete-frac", type=float, default=DEFAULT_MAX_DELETE_FRAC)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.dsn:
        log.error("no DSN: set PG_CONNECTION_STRING or pass --dsn")
        return 2
    if args.desired_json is None and args.congress is None:
        log.error("need --congress (corpus walk) or --desired-json (drill)")
        return 2

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
              + f"-{socket.gethostname()}-{os.urandom(3).hex()}")

    try:
        chunks = (desired_from_json(Path(args.desired_json)) if args.desired_json
                  else desired_from_corpus(Path(args.corpus_root), args.congress))
        desired = index_desired(chunks)
        congress = args.congress or int(next(iter(next(iter(desired.values())).values()))["congress"])
    except ReconcileError as e:
        log.error(str(e))
        return 1

    conn = psycopg2.connect(args.dsn)
    try:
        try:
            with conn.cursor() as cur:
                schema = resolve_schema(cur)
                actual = fetch_actual(cur, schema, congress)
            log.info(f"target {schema}.chunks | congress {congress} | "
                     f"desired: {len(desired)} bills / {sum(len(c) for c in desired.values())} chunks | "
                     f"actual: {len(actual)} bills / {sum(len(s) for s in actual.values())} chunks")

            plan = build_plan(desired, actual, args.max_delete_frac)
        except ReconcileError as e:
            # Plan-stage abort: nothing has been mutated and no audit row was
            # opened — refuse loudly and cleanly.
            log.error(f"reconcile REFUSED: {e}")
            return 1
        log.info(f"plan: replace {len(plan['replace'])} bill(s), "
                 f"delete {len(plan['delete'])} bill(s), "
                 f"embed {len(plan['new_hashes'])} new chunk(s)")

        if args.dry_run:
            log.info("[DRY RUN] stopping before execution")
            return 0

        try:
            audit_start(args.dsn, run_id, congress, args.model)
        except Exception as e:
            log.error(f"could not open audit row (nothing mutated): {e}")
            return 1
        try:
            with conn.cursor() as cur:
                reusable = load_reusable_embeddings(cur, schema, plan["replace"])
            new_chunks = {h: c for uid in plan["replace"]
                          for h, c in desired[uid].items() if h in plan["new_hashes"]}
            embeddings = embed_new(new_chunks, args.backend, args.model, args.dimensions)

            executed = execute_plan(conn, schema, plan, desired, embeddings,
                                    reusable, args.model)
            with conn.cursor() as cur:
                total = verify(cur, schema, congress, desired)
                cur.execute("SELECT ops.refresh_data_versions()")
            conn.commit()
        except (ReconcileError, Exception) as e:
            conn.rollback()
            audit_finish(args.dsn, run_id, "failed", {"error": str(e)[:2000]})
            log.error(f"reconcile FAILED: {e}")
            return 1

        audit_finish(args.dsn, run_id, "success", {
            "planned_add": len(plan["new_hashes"]),
            "planned_update": sum(len(desired[u]) for u in plan["replace"]) - len(plan["new_hashes"]),
            "planned_delete": len(plan["delete"]),
            "executed_add": executed["add"],
            "executed_update": executed["update"],
            "executed_delete": executed["delete"],
            "corpus_bill_count": len(desired),
            "corpus_chunk_count": total,
        })
        log.info(f"reconcile OK: {executed['add']} added, {executed['update']} reused, "
                 f"{executed['delete']} deleted rows | corpus now {total} chunks")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
