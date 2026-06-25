#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg2-binary"]
# ///
"""Retrieval evaluation harness for CSearch semantic search.

Runs the queries in eval_set.json against the live nlp.* tables and reports
recall@k, precision@k, MRR, and latency p50/p95 per category and overall. This
is the measurement layer the criticisms call for: without it you cannot tell
whether a change to chunking, the embedding model, HNSW settings, or rank
fusion improved or regressed retrieval.

Providers:
  --provider openai   real embeddings (needs OPENAI_API_KEY); the production path
  --provider fake     deterministic smoke provider for CI against db/seed fixtures

Modes:
  --mode vector       vector top-k only (current production behavior)
  --mode keyword      full-text only (search_bills)
  --mode hybrid       Reciprocal Rank Fusion of keyword + vector

Examples:
  uv run backend/nlp/eval/run_eval.py --provider fake --only-fixtures \
      --dsn postgresql://postgres:postgres@localhost:55432/csearch
  OPENAI_API_KEY=sk-... uv run backend/nlp/eval/run_eval.py --mode hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import psycopg2

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
RRF_K = 60


# --- Embedding providers -----------------------------------------------------

def embed_openai(query: str) -> list[float]:
    from openai import OpenAI  # imported lazily so --provider fake needs no key

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query], dimensions=1536)
    return resp.data[0].embedding


def embed_fake(query: str) -> list[float]:
    """Deterministic smoke provider aligned with db/seed/fixtures.sql.

    Maps broadband-ish queries to the axis-1 unit vector (fixture hr42) and
    water-ish queries to the axis-2 unit vector (fixture s100). Everything else
    is the zero vector. Only meaningful for the fixture corpus.
    """
    q = query.lower()
    vec = [0.0] * 1536
    if any(w in q for w in ("broadband", "internet", "rural", "jane doe")):
        vec[0] = 1.0
    elif any(w in q for w in ("water", "clean", "environment")):
        vec[1] = 1.0
    return vec


PROVIDERS = {"openai": embed_openai, "fake": embed_fake}


# --- Retrieval ---------------------------------------------------------------

def vector_search(cur, vector: list[float], k: int) -> list[str]:
    literal = "[" + ",".join(f"{v:.6g}" for v in vector) + "]"
    cur.execute(
        """
        SELECT c.bill_id
        FROM (
            SELECT chunk_id, embedding <=> %s::vector AS dist
            FROM nlp.bill_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ) tk
        JOIN nlp.bill_chunks c ON c.id = tk.chunk_id
        GROUP BY c.bill_id
        ORDER BY min(tk.dist)
        LIMIT %s
        """,
        (literal, literal, k * 5, k),
    )
    return [r[0] for r in cur.fetchall()]


def keyword_search(cur, query: str, k: int) -> list[str]:
    cur.execute(
        "SELECT billtype, billnumber, congress FROM search_bills(%s, NULL, NULL, %s)",
        (query, k),
    )
    return [f"{bt}{bn}-{cg}" for bt, bn, cg in cur.fetchall()]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def retrieve(cur, query: str, mode: str, provider, k: int) -> list[str]:
    if mode == "keyword":
        return keyword_search(cur, query, k)
    vec = provider(query)
    if mode == "vector":
        return vector_search(cur, vec, k)
    if mode == "hybrid":
        return rrf([keyword_search(cur, query, k), vector_search(cur, vec, k)])[:k]
    raise ValueError(f"unknown mode {mode}")


# --- Metrics -----------------------------------------------------------------

def metrics_for(retrieved: list[str], expected: list[str], k: int) -> tuple[float, float, float]:
    if not expected:
        return (float("nan"), float("nan"), float("nan"))
    top = retrieved[:k]
    hits = [b for b in top if b in expected]
    recall = len(set(hits)) / len(set(expected))
    precision = len(hits) / max(len(top), 1)
    mrr = 0.0
    for rank, b in enumerate(top, 1):
        if b in expected:
            mrr = 1.0 / rank
            break
    return recall, precision, mrr


def main() -> int:
    parser = argparse.ArgumentParser(description="CSearch retrieval evaluation")
    parser.add_argument("--dsn", default=os.environ.get("PG_CONNECTION_STRING"))
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="vector")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--only-fixtures", action="store_true", help="run only fixture-flagged queries")
    args = parser.parse_args()

    if not args.dsn:
        print("Provide --dsn or PG_CONNECTION_STRING", flush=True)
        return 2

    data = json.loads(EVAL_SET.read_text())
    queries = [q for q in data["queries"] if q.get("expected_bill_ids")]
    if args.only_fixtures:
        queries = [q for q in queries if q.get("fixture")]

    provider = PROVIDERS[args.provider]
    conn = psycopg2.connect(args.dsn)
    by_category: dict[str, list[tuple[float, float, float]]] = {}
    latencies: list[float] = []
    try:
        with conn.cursor() as cur:
            for q in queries:
                started = time.perf_counter()
                retrieved = retrieve(cur, q["query"], args.mode, provider, args.k)
                latencies.append((time.perf_counter() - started) * 1000)
                m = metrics_for(retrieved, q["expected_bill_ids"], args.k)
                by_category.setdefault(q["category"], []).append(m)
    finally:
        conn.close()

    print(f"\nEval: mode={args.mode} provider={args.provider} k={args.k} queries={len(queries)}\n")
    print(f"{'category':<34}{'recall@k':>10}{'prec@k':>10}{'MRR':>8}")
    all_metrics: list[tuple[float, float, float]] = []
    for category, rows in sorted(by_category.items()):
        all_metrics.extend(rows)
        rec = statistics.mean(r[0] for r in rows)
        prec = statistics.mean(r[1] for r in rows)
        mrr = statistics.mean(r[2] for r in rows)
        print(f"{category:<34}{rec:>10.3f}{prec:>10.3f}{mrr:>8.3f}")
    if all_metrics:
        print(f"{'OVERALL':<34}"
              f"{statistics.mean(r[0] for r in all_metrics):>10.3f}"
              f"{statistics.mean(r[1] for r in all_metrics):>10.3f}"
              f"{statistics.mean(r[2] for r in all_metrics):>8.3f}")
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"\nlatency p50={p50:.1f}ms p95={p95:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
