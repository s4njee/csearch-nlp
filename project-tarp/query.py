#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "psycopg2-binary",
# ]
# ///
"""
query.py — Run semantic search against embedded bill chunks in PostgreSQL pgvector.

Embeds a natural-language query with OpenAI or Ollama, searches
`nlp.bill_embeddings` joined to `nlp.bill_chunks`, prints the top matches, and
can optionally ask an OpenAI LLM to synthesize an answer grounded in the
retrieved chunks.

Usage:
    python query.py
    python query.py "financial crisis bailout bills"
    python query.py --backend ollama --embed-model qwen3-embedding:8b-q8_0 --no-answer
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import psycopg2

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DSN = os.environ.get("PG_CONNECTION_STRING", "")
DEFAULT_SCHEMA = "nlp"
DEFAULT_CHUNK_TABLE = "bill_chunks"
DEFAULT_EMBEDDING_TABLE = "bill_embeddings"
DEFAULT_BACKEND = "openai"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_EMBED_DIMENSIONS = None
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QUERY_PREFIX = "Represent this query for retrieving relevant legislative passages: "
DEFAULT_ANSWER_MODEL = "gpt-5.4-nano"
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_LEN = 400
DEFAULT_CONGRESS_MIN = 0
DEFAULT_CONGRESS_MAX = 999

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("query")


@dataclass
class SearchHit:
    bill_id: str
    section_enum: str
    section_header: str
    title: str
    status: str
    body: str
    similarity: float
    congress: int
    chunk_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def openai_embed_query(client: OpenAI, query: str, model: str, dimensions: Optional[int]) -> list[float]:
    kwargs = {
        "input": [query],
        "model": model,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return response.data[0].embedding


def ollama_embed_query(base_url: str, query: str, model: str) -> list[float]:
    payload = json.dumps({
        "model": model,
        "input": query,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc.reason}") from exc

    data = json.loads(body)
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        return embeddings[0]
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (float, int)):
        return embeddings
    raise RuntimeError(f"Unexpected Ollama response: {data}")


def embed_query(client, backend: str, query: str, model: str, dimensions: Optional[int], ollama_url: str, query_prefix: str) -> list[float]:
    if backend == "openai":
        return openai_embed_query(client, query, model, dimensions)
    if backend == "ollama":
        prefixed = f"{query_prefix}{query}" if query_prefix else query
        return ollama_embed_query(ollama_url, prefixed, model)
    raise ValueError(f"Unsupported backend: {backend}")


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.17g}" for v in values) + "]"


def snippet(text: str, max_len: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def format_hit(hit: SearchHit, max_len: int) -> str:
    header = hit.section_header or "(no header)"
    section_enum = hit.section_enum or "?"
    title_line = f" | {hit.title}" if hit.title else ""
    return (
        f"[{hit.similarity:.4f}] {hit.bill_id} §{section_enum} — {header}{title_line}\n"
        f"{snippet(hit.body, max_len)}"
    )


def build_answer_context(hits: list[SearchHit], max_len: int) -> str:
    parts = []
    for idx, hit in enumerate(hits, 1):
        parts.append(
            "\n".join([
                f"Result {idx}",
                f"Bill: {hit.bill_id}",
                f"Title: {hit.title}",
                f"Status: {hit.status}",
                f"Congress: {hit.congress}",
                f"Section: {hit.section_enum or '?'} — {hit.section_header or ''}",
                f"Text: {snippet(hit.body, max_len)}",
            ])
        )
    return "\n\n".join(parts)


def generate_answer(client: OpenAI, model: str, query: str, hits: list[SearchHit], max_len: int) -> str:
    context = build_answer_context(hits, max_len)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided congressional bill excerpts. "
                    "Cite specific bill numbers when possible, mention uncertainty when the excerpts are insufficient, "
                    "and do not fabricate facts."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}",
            },
        ],
    )
    return response.output_text.strip()


def pgvector_search(
    dsn: str,
    schema: str,
    chunk_table: str,
    embedding_table: str,
    vector: list[float],
    limit: int,
    congress_min: int,
    congress_max: int,
) -> list[SearchHit]:
    vector_sql = vector_literal(vector)
    sql = f"""
        SELECT
          c.bill_id,
          c.section_enum,
          c.section_header,
          c.title,
          c.status,
          c.body,
          1 - (e.embedding <=> %s::vector) AS similarity,
          c.congress,
          c.chunk_type
        FROM {schema}.{embedding_table} e
        JOIN {schema}.{chunk_table} c ON c.id = e.chunk_id
        WHERE c.congress BETWEEN %s AND %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_sql, congress_min, congress_max, vector_sql, limit))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        SearchHit(
            bill_id=row[0] or "?",
            section_enum=row[1] or "",
            section_header=row[2] or "",
            title=row[3] or "",
            status=row[4] or "",
            body=row[5] or "",
            similarity=float(row[6] or 0.0),
            congress=int(row[7] or 0),
            chunk_type=row[8] or "",
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic search over Project TARP bill chunks in pgvector")
    parser.add_argument("query", nargs="*", help="Natural-language query text")
    parser.add_argument("--dsn", type=str, default=DEFAULT_DSN,
                        help="PostgreSQL connection string (default: PG_CONNECTION_STRING env var)")
    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA,
                        help=f"Target schema (default: {DEFAULT_SCHEMA})")
    parser.add_argument("--chunk-table", type=str, default=DEFAULT_CHUNK_TABLE,
                        help=f"Chunk table name (default: {DEFAULT_CHUNK_TABLE})")
    parser.add_argument("--embedding-table", type=str, default=DEFAULT_EMBEDDING_TABLE,
                        help=f"Embedding table name (default: {DEFAULT_EMBEDDING_TABLE})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of search hits to return (default: {DEFAULT_TOP_K})")
    parser.add_argument("--congress-min", type=int, default=DEFAULT_CONGRESS_MIN,
                        help=f"Minimum congress filter (default: {DEFAULT_CONGRESS_MIN})")
    parser.add_argument("--congress-max", type=int, default=DEFAULT_CONGRESS_MAX,
                        help=f"Maximum congress filter (default: {DEFAULT_CONGRESS_MAX})")
    parser.add_argument("--backend", choices=["openai", "ollama"], default=os.environ.get("EMBEDDING_BACKEND", DEFAULT_BACKEND),
                        help=f"Embedding backend (default: {os.environ.get('EMBEDDING_BACKEND', DEFAULT_BACKEND)})")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="Embedding model name; defaults depend on backend")
    parser.add_argument("--dimensions", type=int, default=None,
                        help="Requested embedding dimensions for backends that support it")
    parser.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
                        help=f"Ollama base URL (default: {os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_URL)})")
    parser.add_argument("--query-prefix", type=str, default=os.environ.get("EMBEDDING_QUERY_PROMPT", DEFAULT_QUERY_PREFIX),
                        help="Optional prefix for query embedding; recommended for local Qwen3 retrieval")
    parser.add_argument("--answer-model", type=str, default=DEFAULT_ANSWER_MODEL,
                        help=f"Answer model (default: {DEFAULT_ANSWER_MODEL})")
    parser.add_argument("--snippet-len", type=int, default=DEFAULT_SNIPPET_LEN,
                        help=f"Displayed snippet length (default: {DEFAULT_SNIPPET_LEN})")
    parser.add_argument("--no-answer", action="store_true",
                        help="Skip answer generation and only show retrieved chunks")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        try:
            query = input("Query: ").strip()
        except EOFError:
            query = ""

    if not query:
        log.error("No query provided.")
        return

    if not args.dsn:
        log.error("PostgreSQL DSN not provided. Set PG_CONNECTION_STRING or pass --dsn.")
        return

    if args.embed_model is None:
        args.embed_model = DEFAULT_EMBED_MODEL if args.backend == "openai" else DEFAULT_OLLAMA_EMBED_MODEL
    if args.dimensions is None and args.backend == "openai":
        args.dimensions = 1536 if args.embed_model == "text-embedding-3-small" else DEFAULT_EMBED_DIMENSIONS

    embedding_client = None
    llm_client = None
    if args.backend == "openai":
        if OpenAI is None:
            log.error("openai package not installed. Run: pip install openai")
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY environment variable not set")
            return
        embedding_client = OpenAI(api_key=api_key)
        llm_client = embedding_client
    elif not args.no_answer:
        if OpenAI is None:
            log.error("openai package not installed. Run: pip install openai")
            return
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY environment variable not set for answer generation")
            return
        llm_client = OpenAI(api_key=api_key)

    log.info(f"Embedding query with backend={args.backend} model={args.embed_model}")
    vector = embed_query(
        embedding_client,
        args.backend,
        query,
        args.embed_model,
        args.dimensions,
        args.ollama_url,
        args.query_prefix,
    )

    log.info(
        f"Searching {args.schema}.{args.embedding_table} via PostgreSQL "
        f"(top_k={args.top_k}, congress={args.congress_min}-{args.congress_max})"
    )
    hits = pgvector_search(
        dsn=args.dsn,
        schema=args.schema,
        chunk_table=args.chunk_table,
        embedding_table=args.embedding_table,
        vector=vector,
        limit=args.top_k,
        congress_min=args.congress_min,
        congress_max=args.congress_max,
    )

    if not hits:
        print("No results.")
        return

    print("\nTop Matches\n")
    for idx, hit in enumerate(hits, 1):
        print(f"{idx}. {format_hit(hit, args.snippet_len)}\n")

    if args.no_answer:
        return

    log.info(f"Generating answer with {args.answer_model}")
    answer = generate_answer(llm_client, args.answer_model, query, hits, args.snippet_len)
    print("Answer\n")
    print(answer)


if __name__ == "__main__":
    main()
