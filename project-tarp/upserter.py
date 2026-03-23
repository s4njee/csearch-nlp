#!/usr/bin/env python3
"""
upserter.py — Load embedded chunk shards into a Qdrant collection.

Reads embedded JSONL shard files produced by embedder.py, creates a Qdrant
collection if needed, and upserts points in batches.

By default, payloads stay relatively lean. Large alias arrays are omitted
unless explicitly requested.

Requires:
    pip install qdrant-client

Usage:
    python upserter.py
    python upserter.py --dry-run
    python upserter.py --recreate
    python upserter.py --batch-size 256
    python upserter.py --include-aliases
    python upserter.py --host 192.168.1.156 --port 6333
"""

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
    from qdrant_client.http.models import PayloadSchemaType
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_DIR = DATA_DIR / "embedded_chunks"

DEFAULT_COLLECTION = "bill_chunks"
DEFAULT_HOST = "192.168.1.156"
DEFAULT_PORT = 6333
DEFAULT_BATCH_SIZE = 256
DEFAULT_VECTOR_SIZE = 1536
SHARD_NAME_RE = re.compile(r"^shard-\d{5}\.jsonl$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("upserter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_embedded_shards(input_path: Path) -> list[Path]:
    """Return embedded shard files in stable order."""
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*.jsonl")
            if SHARD_NAME_RE.fullmatch(path.name)
        )
    if input_path.is_file() and SHARD_NAME_RE.fullmatch(input_path.name):
        return [input_path]
    raise ValueError(f"Unsupported input path: {input_path}")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from a file."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def point_id(chunk: dict) -> str:
    """Stable UUID point id derived from canonical chunk identity."""
    raw = "|".join([
        str(chunk.get("document_text_hash", "")),
        str(chunk.get("section_text_hash", "")),
        str(chunk.get("chunk_index", 0)),
    ])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def build_payload(chunk: dict, include_aliases: bool) -> dict:
    """Build the Qdrant payload for one embedded chunk."""
    payload = {
        "bill_id": chunk.get("bill_id"),
        "canonical_bill_id": chunk.get("canonical_bill_id"),
        "congress": chunk.get("congress"),
        "type": chunk.get("type"),
        "number": chunk.get("number"),
        "short_title": chunk.get("short_title"),
        "status": chunk.get("status"),
        "version": chunk.get("version"),
        "section_enum": chunk.get("section_enum"),
        "section_header": chunk.get("section_header"),
        "chunk_index": chunk.get("chunk_index"),
        "original_chunk_index": chunk.get("original_chunk_index"),
        "document_text_hash": chunk.get("document_text_hash"),
        "section_text_hash": chunk.get("section_text_hash"),
        "text": chunk.get("text"),
        "tokens": chunk.get("tokens"),
    }
    if include_aliases:
        payload["document_aliases"] = chunk.get("document_aliases", [])
        payload["section_aliases"] = chunk.get("section_aliases", [])
    return payload


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, recreate: bool) -> None:
    """Create or recreate the target collection."""
    existing = {c.name for c in client.get_collections().collections}
    if recreate and collection in existing:
        log.info(f"Deleting existing collection: {collection}")
        client.delete_collection(collection_name=collection)
        existing.remove(collection)

    if collection not in existing:
        log.info(f"Creating collection {collection} (size={vector_size}, distance=Cosine)")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        info = client.get_collection(collection_name=collection)
        actual_size = info.config.params.vectors.size
        if actual_size != vector_size:
            raise RuntimeError(
                f"Collection {collection} has vector size {actual_size}, expected {vector_size}. "
                "Use --recreate or a different collection name."
            )


def ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Create useful payload indexes if they do not already exist."""
    index_specs = {
        "bill_id": PayloadSchemaType.KEYWORD,
        "canonical_bill_id": PayloadSchemaType.KEYWORD,
        "congress": PayloadSchemaType.INTEGER,
        "type": PayloadSchemaType.KEYWORD,
        "number": PayloadSchemaType.KEYWORD,
        "section_enum": PayloadSchemaType.KEYWORD,
        "document_text_hash": PayloadSchemaType.KEYWORD,
        "section_text_hash": PayloadSchemaType.KEYWORD,
    }
    for field_name, schema in index_specs.items():
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            # Qdrant treats duplicate index creation as an error in some versions.
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upsert embedded chunk shards into Qdrant")
    parser.add_argument("--input", type=str, default=str(INPUT_DIR),
                        help=f"Input embedded shard directory (default: {INPUT_DIR})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help=f"Qdrant host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Qdrant port (default: {DEFAULT_PORT})")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION,
                        help=f"Collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE,
                        help=f"Vector size (default: {DEFAULT_VECTOR_SIZE})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Points per upsert call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--include-aliases", action="store_true",
                        help="Include document_aliases and section_aliases in payloads")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the collection before loading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be upserted without touching Qdrant")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input path not found: {input_path}")
        return

    log.info(f"Starting upserter | input={input_path} | host={args.host}:{args.port} | collection={args.collection}")

    try:
        shard_paths = iter_embedded_shards(input_path)
    except ValueError as e:
        log.error(str(e))
        return

    if not shard_paths:
        log.error(f"No embedded shard files found under {input_path}")
        return

    log.info(f"Found {len(shard_paths)} embedded shard file(s); scanning counts and embeddings...")

    total_points = 0
    total_tokens = 0
    for shard_index, shard_path in enumerate(shard_paths, 1):
        log.info(f"Scanning shard {shard_index}/{len(shard_paths)}: {shard_path.name}")
        shard_points = 0
        shard_tokens = 0
        for chunk in read_jsonl(shard_path):
            if "embedding" not in chunk:
                raise RuntimeError(f"Missing embedding in shard {shard_path}")
            total_points += 1
            total_tokens += int(chunk.get("tokens", 0))
            shard_points += 1
            shard_tokens += int(chunk.get("tokens", 0))
        log.info(
            f"  Shard {shard_index}/{len(shard_paths)} scan complete: "
            f"{shard_points} points, {shard_tokens:,} tokens"
        )

    log.info(f"Discovered {len(shard_paths)} shard(s) with {total_points} embedded chunk(s)")
    log.info(f"Target collection: {args.collection} on {args.host}:{args.port}")

    if args.dry_run:
        log.info("[DRY RUN] Exiting without connecting to Qdrant")
        return

    client = QdrantClient(host=args.host, port=args.port)
    ensure_collection(client, args.collection, args.vector_size, args.recreate)
    ensure_payload_indexes(client, args.collection)

    upserted = 0
    for shard_index, shard_path in enumerate(shard_paths, 1):
        chunks = read_jsonl(shard_path)
        log.info(f"Shard {shard_index}/{len(shard_paths)}: {shard_path.name} ({len(chunks)} points)")

        batch = []
        for chunk in chunks:
            batch.append(
                PointStruct(
                    id=point_id(chunk),
                    vector=chunk["embedding"],
                    payload=build_payload(chunk, include_aliases=args.include_aliases),
                )
            )

            if len(batch) >= args.batch_size:
                client.upsert(collection_name=args.collection, points=batch, wait=True)
                upserted += len(batch)
                log.info(f"  Upserted {upserted}/{total_points} points")
                batch = []

        if batch:
            client.upsert(collection_name=args.collection, points=batch, wait=True)
            upserted += len(batch)
            log.info(f"  Upserted {upserted}/{total_points} points")

    info = client.get_collection(collection_name=args.collection)
    log.info(f"{'=' * 60}")
    log.info(f"DONE: {upserted} points upserted")
    log.info(f"Collection count reported by Qdrant: {info.points_count}")


if __name__ == "__main__":
    main()
