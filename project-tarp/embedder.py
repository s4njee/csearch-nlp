#!/usr/bin/env python3
"""
embedder.py — Convert text chunks into vector embeddings via OpenAI API.

Reads processed JSONL shard files, batches the text, calls the OpenAI
embeddings API, and writes embedded JSONL shards that mirror the input layout.

Supports incremental embedding: if an output shard already exists, only
missing/unembedded chunks in that shard are sent to the API.

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python embedder.py                             # embed all chunk shards
    python embedder.py --batch-size 500            # smaller batches
    python embedder.py --delay-seconds 3           # slower pace between batches
    python embedder.py --checkpoint-every 10       # save every 10 batches
    python embedder.py --dry-run                   # show cost estimate only
"""

import argparse
import json
import logging
import os
import re
import sys
import time
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
INPUT_FILE = DATA_DIR / "processed_chunks"
OUTPUT_FILE = DATA_DIR / "embedded_chunks"

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 500
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHECKPOINT_EVERY = 10

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
SHARD_NAME_RE = re.compile(r"^shard-\d{5}\.jsonl$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_batch(client: OpenAI, texts: list[str], model: str, dimensions: int) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Returns list of vectors."""
    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from a file."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL records atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")
    tmp_path.replace(path)
    log.info(f"Checkpoint saved: {path} ({len(records)} chunks)")


def chunk_identity(chunk: dict) -> tuple[str, str, int]:
    """Stable identity for incremental embedding after dedup."""
    document_hash = chunk.get("document_text_hash")
    section_hash = chunk.get("section_text_hash")
    if document_hash or section_hash:
        return (
            document_hash or "",
            section_hash or "",
            int(chunk.get("chunk_index", 0)),
        )
    return (
        chunk.get("bill_id", ""),
        chunk.get("section_enum", ""),
        int(chunk.get("chunk_index", 0)),
    )


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection for 429/rate-limit responses across client versions."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in text


def iter_input_shards(input_path: Path) -> list[Path]:
    """Return all input shard files in stable order."""
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*.jsonl")
            if SHARD_NAME_RE.fullmatch(path.name)
        )
    if input_path.is_file() and SHARD_NAME_RE.fullmatch(input_path.name):
        return [input_path]
    raise ValueError(f"Unsupported input path for shard embedding: {input_path}")


def shard_output_path(input_root: Path, output_root: Path, shard_path: Path) -> Path:
    """Map an input shard to its mirrored output shard path."""
    if input_root.is_dir():
        return output_root / shard_path.relative_to(input_root)
    return output_root / shard_path.name


def load_existing_shard(path: Path) -> dict[tuple[str, str, int], dict]:
    """Load embedded records from an output shard indexed by chunk identity."""
    if not path.exists():
        return {}
    existing = {}
    for record in read_jsonl(path):
        if "embedding" in record:
            existing[chunk_identity(record)] = record
    return existing


def build_shard_records(input_chunks: list[dict], embedded_by_key: dict[tuple[str, str, int], dict]) -> list[dict]:
    """Rebuild shard output in input order using the latest embedded records."""
    records = []
    for chunk in input_chunks:
        key = chunk_identity(chunk)
        if key in embedded_by_key:
            records.append(embedded_by_key[key])
    return records


def save_manifest(output_root: Path, summary: dict) -> None:
    """Save a lightweight run manifest for the embedded shard directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Manifest saved: {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed text chunks via OpenAI API")
    parser.add_argument("--input", type=str, default=str(INPUT_FILE),
                        help=f"Input chunk shard directory (default: {INPUT_FILE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help=f"Output embedded shard directory (default: {OUTPUT_FILE})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS,
                        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Texts per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS,
                        help=f"Seconds to wait between successful batches (default: {DEFAULT_DELAY_SECONDS})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"Max retries for rate-limited batches (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help=f"Save checkpoint every N successful batches within a shard (default: {DEFAULT_CHECKPOINT_EVERY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without calling the API")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error(f"Input path not found: {input_path}")
        log.error("Run chunker.py first to generate processed chunk shards")
        return

    try:
        shard_paths = iter_input_shards(input_path)
    except ValueError as e:
        log.error(str(e))
        return

    if not shard_paths:
        log.error(f"No input shard files found under {input_path}")
        return

    total_chunks = 0
    total_tokens = 0
    total_already_done = 0
    shard_statuses = []

    for shard_path in shard_paths:
        input_chunks = read_jsonl(shard_path)
        existing_by_key = load_existing_shard(shard_output_path(input_path, output_path, shard_path))
        shard_total = len(input_chunks)
        shard_pending = [chunk for chunk in input_chunks if chunk_identity(chunk) not in existing_by_key]
        shard_pending_tokens = sum(chunk["tokens"] for chunk in shard_pending)

        total_chunks += shard_total
        total_tokens += shard_pending_tokens
        total_already_done += shard_total - len(shard_pending)
        shard_statuses.append({
            "input_shard": str(shard_path),
            "output_shard": str(shard_output_path(input_path, output_path, shard_path)),
            "chunk_count": shard_total,
            "already_embedded": shard_total - len(shard_pending),
            "to_embed": len(shard_pending),
        })

    if total_chunks == 0:
        log.info("No chunks found. Nothing to do.")
        return

    log.info(f"Discovered {len(shard_paths)} shard(s) with {total_chunks} chunk(s) total")
    log.info(f"Need to embed: {total_chunks - total_already_done} chunks ({total_already_done} already done)")

    price_per_m = PRICING.get(args.model, 0.02)
    est_cost = total_tokens / 1_000_000 * price_per_m
    log.info(f"Estimated cost: {total_tokens:,} tokens × ${price_per_m}/1M = ${est_cost:.4f}")

    if args.dry_run:
        log.info("[DRY RUN] Exiting without calling API")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    total_embedded_now = 0
    current_output_shard = None
    current_input_chunks = None
    current_embedded_by_key = None
    try:
        for shard_index, shard_path in enumerate(shard_paths, 1):
            output_shard = shard_output_path(input_path, output_path, shard_path)
            input_chunks = read_jsonl(shard_path)
            embedded_by_key = load_existing_shard(output_shard)
            current_output_shard = output_shard
            current_input_chunks = input_chunks
            current_embedded_by_key = embedded_by_key

            to_embed = [chunk for chunk in input_chunks if chunk_identity(chunk) not in embedded_by_key]
            if not to_embed:
                log.info(f"Shard {shard_index}/{len(shard_paths)} already complete: {output_shard}")
                continue

            batches = [to_embed[i:i + args.batch_size] for i in range(0, len(to_embed), args.batch_size)]
            log.info(
                f"Shard {shard_index}/{len(shard_paths)}: {shard_path.name} | "
                f"{len(to_embed)}/{len(input_chunks)} chunks to embed | "
                f"{len(batches)} batch(es)"
            )
            log.info(
                f"Inter-batch delay: {args.delay_seconds:.1f}s | "
                f"Max rate-limit retries: {args.max_retries} | "
                f"Checkpoint every {args.checkpoint_every} batch(es)"
            )

            shard_embedded_now = 0
            for bi, batch in enumerate(batches, 1):
                texts = [chunk["text"] for chunk in batch]

                for attempt in range(args.max_retries + 1):
                    try:
                        vectors = embed_batch(client, texts, args.model, args.dimensions)
                        break
                    except Exception as e:
                        if is_rate_limit_error(e) and attempt < args.max_retries:
                            backoff = args.delay_seconds * (2 ** attempt)
                            log.warning(
                                f"Rate limited on shard {shard_index} batch {bi} "
                                f"(attempt {attempt + 1}/{args.max_retries + 1}); "
                                f"sleeping {backoff:.1f}s before retry"
                            )
                            time.sleep(backoff)
                            continue

                        log.error(f"API error on shard {shard_index} batch {bi}: {e}")
                        write_jsonl(output_shard, build_shard_records(input_chunks, embedded_by_key))
                        _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
                        return

                for chunk, vector in zip(batch, vectors):
                    updated = dict(chunk)
                    updated["embedding"] = vector
                    embedded_by_key[chunk_identity(updated)] = updated
                    shard_embedded_now += 1
                    total_embedded_now += 1

                batch_tokens = sum(chunk["tokens"] for chunk in batch)
                log.info(
                    f"Shard {shard_index}/{len(shard_paths)} batch {bi}/{len(batches)}: "
                    f"{len(batch)} chunks, {batch_tokens:,} tokens — done "
                    f"({shard_embedded_now}/{len(to_embed)} in shard)"
                )

                if args.checkpoint_every > 0 and bi % args.checkpoint_every == 0:
                    write_jsonl(output_shard, build_shard_records(input_chunks, embedded_by_key))

                if bi < len(batches):
                    time.sleep(args.delay_seconds)

            write_jsonl(output_shard, build_shard_records(input_chunks, embedded_by_key))

    except KeyboardInterrupt:
        if current_output_shard is not None and current_input_chunks is not None and current_embedded_by_key is not None:
            write_jsonl(current_output_shard, build_shard_records(current_input_chunks, current_embedded_by_key))
        log.warning("Interrupted by user; partial shard progress has been checkpointed")
        _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
        return

    _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
    log.info(f"{'=' * 60}")
    log.info(f"DONE: {total_embedded_now} chunks embedded in this run")
    log.info(f"Saved embedded shards under {output_path}")


def _save_final_manifest(output_root: Path, args, shard_statuses: list[dict], embedded_now: int) -> None:
    """Save a lightweight manifest describing the embedded shard directory."""
    summary = {
        "model": args.model,
        "dimensions": args.dimensions,
        "batch_size": args.batch_size,
        "delay_seconds": args.delay_seconds,
        "max_retries": args.max_retries,
        "checkpoint_every": args.checkpoint_every,
        "embedded_in_this_run": embedded_now,
        "shards": shard_statuses,
    }
    save_manifest(output_root, summary)


if __name__ == "__main__":
    main()
