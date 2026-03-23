#!/usr/bin/env python3
"""
query.py — Run semantic search against embedded bill chunks in Qdrant.

Embeds a natural-language query with OpenAI, searches the `bill_chunks`
collection in Qdrant, prints the top matches, and can optionally ask an LLM
to synthesize an answer grounded in the retrieved chunks.

Requires:
    pip install openai qdrant-client
    export OPENAI_API_KEY=sk-...

Usage:
    python query.py
    python query.py "financial crisis bailout bills"
    python query.py --top-k 8 --no-answer
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_QDRANT_HOST = "192.168.1.156"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_COLLECTION = "bill_chunks"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_DIMENSIONS = 1536
DEFAULT_ANSWER_MODEL = "gpt-5.4-nano"
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_LEN = 400

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("query")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_query(client: OpenAI, query: str, model: str, dimensions: int) -> list[float]:
    """Embed a single natural-language query."""
    response = client.embeddings.create(
        input=[query],
        model=model,
        dimensions=dimensions,
    )
    return response.data[0].embedding


def snippet(text: str, max_len: int) -> str:
    """Truncate long chunk text for terminal display."""
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def format_hit(hit, max_len: int) -> str:
    """Render a readable search hit."""
    payload = hit.payload or {}
    bill_id = payload.get("bill_id", "?")
    header = payload.get("section_header") or "(no header)"
    section_enum = payload.get("section_enum") or "?"
    title = payload.get("short_title") or ""
    score = getattr(hit, "score", None)
    score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
    body = snippet(payload.get("text", ""), max_len)
    title_line = f" | {title}" if title else ""
    return (
        f"[{score_text}] {bill_id} §{section_enum} — {header}{title_line}\n"
        f"{body}"
    )


def build_answer_context(hits: list, max_len: int) -> str:
    """Build a compact grounded context block for answer generation."""
    parts = []
    for idx, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        parts.append(
            "\n".join([
                f"Result {idx}",
                f"Bill: {payload.get('bill_id', '?')}",
                f"Title: {payload.get('short_title', '')}",
                f"Status: {payload.get('status', '')}",
                f"Section: {payload.get('section_enum', '?')} — {payload.get('section_header', '')}",
                f"Text: {snippet(payload.get('text', ''), max_len)}",
            ])
        )
    return "\n\n".join(parts)


def generate_answer(client: OpenAI, model: str, query: str, hits: list, max_len: int) -> str:
    """Ask the LLM for a grounded answer using retrieved chunk context."""
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


def qdrant_search(client: QdrantClient, collection: str, vector: list[float], limit: int):
    """Support both older .search() and newer .query_points() client APIs."""
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return getattr(result, "points", result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic search over Project TARP bill chunks")
    parser.add_argument("query", nargs="*", help="Natural-language query text")
    parser.add_argument("--host", type=str, default=DEFAULT_QDRANT_HOST,
                        help=f"Qdrant host (default: {DEFAULT_QDRANT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_QDRANT_PORT,
                        help=f"Qdrant port (default: {DEFAULT_QDRANT_PORT})")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of search hits to return (default: {DEFAULT_TOP_K})")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                        help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--dimensions", type=int, default=DEFAULT_EMBED_DIMENSIONS,
                        help=f"Embedding dimensions (default: {DEFAULT_EMBED_DIMENSIONS})")
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set")
        return

    openai_client = OpenAI(api_key=api_key)
    qdrant = QdrantClient(host=args.host, port=args.port)

    log.info(f"Embedding query with {args.embed_model}")
    vector = embed_query(openai_client, query, args.embed_model, args.dimensions)

    log.info(f"Searching {args.collection} on {args.host}:{args.port} (top_k={args.top_k})")
    hits = qdrant_search(qdrant, args.collection, vector, args.top_k)

    if not hits:
        print("No results.")
        return

    print("\nTop Matches\n")
    for idx, hit in enumerate(hits, 1):
        print(f"{idx}. {format_hit(hit, args.snippet_len)}\n")

    if args.no_answer:
        return

    log.info(f"Generating answer with {args.answer_model}")
    answer = generate_answer(openai_client, args.answer_model, query, hits, args.snippet_len)
    print("Answer\n")
    print(answer)


if __name__ == "__main__":
    main()
