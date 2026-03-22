#!/usr/bin/env python3
"""
embedder.py — Convert text chunks into vector embeddings via OpenAI API.

Reads processed_chunks.json, batches the text, calls the OpenAI embeddings
API, and saves the result as embedded_chunks.json (a checkpoint file so the
API call doesn't need to be re-run during downstream debugging).

Supports incremental embedding: if embedded_chunks.json already exists,
only new/unembedded chunks are sent to the API.

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python embedder.py                             # embed all chunks
    python embedder.py --model text-embedding-3-large --dimensions 1536
    python embedder.py --batch-size 500            # smaller batches
    python embedder.py --dry-run                   # show cost estimate only
"""

import argparse
import json
import time
import logging
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "processed_chunks.json"
OUTPUT_FILE = DATA_DIR / "embedded_chunks.json"

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 500  # OpenAI supports up to 2048 inputs per request

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embedder")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(client: OpenAI, texts: list[str], model: str, dimensions: int) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Returns list of vectors."""
    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions,
    )
    # Response items are in same order as input
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(description="Embed text chunks via OpenAI API")
    parser.add_argument("--input", type=str, default=str(INPUT_FILE),
                        help=f"Input chunks file (default: {INPUT_FILE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help=f"Output checkpoint file (default: {OUTPUT_FILE})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS,
                        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Texts per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without calling the API")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load chunks
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        log.error("Run chunker.py first to generate processed_chunks.json")
        return
    chunks = json.loads(input_path.read_text())
    log.info(f"Loaded {len(chunks)} chunks from {input_path.name}")

    # Load existing checkpoint for incremental embedding
    existing = {}
    if output_path.exists():
        existing_chunks = json.loads(output_path.read_text())
        # Index by (bill_id, section_enum, chunk_index) to detect already-embedded
        for ec in existing_chunks:
            key = (ec["bill_id"], ec.get("section_enum", ""), ec.get("chunk_index", 0))
            if "embedding" in ec:
                existing[key] = ec
        log.info(f"Found {len(existing)} already-embedded chunks in checkpoint")

    # Determine which chunks need embedding
    to_embed = []
    already_done = []
    for c in chunks:
        key = (c["bill_id"], c.get("section_enum", ""), c.get("chunk_index", 0))
        if key in existing:
            already_done.append(existing[key])
        else:
            to_embed.append(c)

    if not to_embed:
        log.info("All chunks already embedded. Nothing to do.")
        return

    log.info(f"Need to embed: {len(to_embed)} chunks ({len(already_done)} already done)")

    # Cost estimate
    total_tokens = sum(c["tokens"] for c in to_embed)
    price_per_m = PRICING.get(args.model, 0.02)
    est_cost = total_tokens / 1_000_000 * price_per_m
    log.info(f"Estimated cost: {total_tokens:,} tokens × ${price_per_m}/1M = ${est_cost:.4f}")

    if args.dry_run:
        log.info("[DRY RUN] Exiting without calling API")
        return

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    # Batch and embed
    batches = [to_embed[i:i + args.batch_size] for i in range(0, len(to_embed), args.batch_size)]
    log.info(f"Processing {len(batches)} batches of up to {args.batch_size}")

    embedded_count = 0
    for bi, batch in enumerate(batches, 1):
        texts = [c["text"] for c in batch]

        try:
            vectors = embed_batch(client, texts, args.model, args.dimensions)
        except Exception as e:
            log.error(f"API error on batch {bi}: {e}")
            log.info(f"Saving checkpoint with {embedded_count + len(already_done)} embedded chunks")
            # Save what we have so far
            _save_checkpoint(output_path, already_done + to_embed[:embedded_count], args)
            return

        # Attach embeddings to chunk records
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
            embedded_count += 1

        batch_tokens = sum(c["tokens"] for c in batch)
        log.info(f"Batch {bi}/{len(batches)}: {len(batch)} chunks, {batch_tokens:,} tokens — done ({embedded_count}/{len(to_embed)})")

        # Rate limit courtesy
        if bi < len(batches):
            time.sleep(0.5)

    # Merge with already-done and save
    all_embedded = already_done + to_embed
    _save_checkpoint(output_path, all_embedded, args)

    log.info(f"{'='*60}")
    log.info(f"DONE: {embedded_count} chunks embedded")
    log.info(f"Total: {len(all_embedded)} chunks in checkpoint")
    log.info(f"Dimensions: {args.dimensions}")
    log.info(f"Saved to {output_path}")


def _save_checkpoint(path: Path, chunks: list[dict], args) -> None:
    """Save embedded chunks with model metadata."""
    output = {
        "model": args.model,
        "dimensions": args.dimensions,
        "count": len(chunks),
        "chunks": chunks,
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info(f"Checkpoint saved: {path} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
