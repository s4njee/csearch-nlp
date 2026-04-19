#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tiktoken",
# ]
# ///
from __future__ import annotations
"""
votes_chunker.py — Turn normalized vote records into embedding-ready chunks.

Most votes become a single chunk. Very long vote descriptions fall back to
sentence-boundary splitting with overlap so the embedding text stays compact
and stable.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from chunker import clean_text, count_tokens, normalize_for_hash, split_with_overlap


INPUT_ROOT = Path(__file__).resolve().parent / "data" / "processed_votes"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "processed_vote_chunks"
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64
DEFAULT_SHARD_SIZE = 2000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("votes_chunker")


def safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def text_value(value) -> str:
    return "" if value is None else str(value)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_congresses(input_root: Path) -> list[int]:
    if not input_root.exists():
        return []
    return [int(path.name) for path in sorted(input_root.iterdir()) if path.is_dir() and path.name.isdigit()]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")))
            f.write("\n")
    tmp_path.replace(path)


def shard_records(records: list[dict], shard_size: int) -> list[list[dict]]:
    if shard_size <= 0:
        return [records]
    return [records[i : i + shard_size] for i in range(0, len(records), shard_size)]


def normalize_vote_pair(question: str, subject: str) -> tuple[str, str]:
    q = clean_text(question)
    s = clean_text(subject)
    if q and s:
        qn = normalize_for_hash(q)
        sn = normalize_for_hash(s)
        if qn.startswith(sn) or sn.startswith(qn):
            return (q, "") if len(q) >= len(s) else ("", s)
    return q, s


def build_vote_chunks(vote: dict, max_tokens: int, overlap: int) -> list[dict]:
    vote_id = text_value(vote.get("vote_id")).strip()
    chamber = text_value(vote.get("chamber")).strip().lower()
    chamber_name = "House" if chamber == "h" else "Senate" if chamber == "s" else chamber.upper() or "Unknown chamber"
    date = text_value(vote.get("date")).strip()
    date_display = date[:10] if len(date) >= 10 else date
    category = clean_text(text_value(vote.get("category")))
    vote_type = clean_text(text_value(vote.get("type")))
    question, subject = normalize_vote_pair(text_value(vote.get("question")), text_value(vote.get("subject")))
    result = clean_text(text_value(vote.get("result")))
    bill_id = text_value(vote.get("bill_id")).strip()

    prefix = clean_text(f"[Vote {vote_id}, {chamber_name}, {date_display}] {category}: {vote_type}")
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if subject:
        parts.append(f"Subject: {subject}")
    if result:
        parts.append(f"Result: {result}")
    if bill_id:
        parts.append(f"Related bill: {bill_id}")

    narrative = "\n".join(parts).strip()
    if not narrative:
        narrative = result or question or subject

    full_text = prefix if not narrative else clean_text(f"{prefix}\n{narrative}")
    full_tokens = count_tokens(full_text)
    if full_tokens <= max_tokens or not narrative:
        content_hash = hash_text(normalize_for_hash(full_text))
        return [{
            "source_hash": hash_text("|".join([vote_id, "0", content_hash])),
            "vote_id": vote_id,
            "congress": safe_int(vote.get("congress")),
            "chamber": chamber,
            "session": text_value(vote.get("session")).strip(),
            "number": safe_int(vote.get("number")),
            "date": date,
            "category": text_value(vote.get("category")).strip(),
            "type": text_value(vote.get("type")).strip(),
            "question": text_value(vote.get("question")).strip(),
            "subject": text_value(vote.get("subject")).strip(),
            "result": text_value(vote.get("result")).strip(),
            "bill_id": bill_id or None,
            "text": full_text,
            "token_count": full_tokens,
            "chunk_index": 0,
            "content_hash": content_hash,
        }]

    prefix_tokens = count_tokens(prefix)
    available = max(1, max_tokens - prefix_tokens - 8)
    chunks = []
    for chunk_index, piece in enumerate(split_with_overlap(narrative, available, overlap)):
        candidate = clean_text(f"{prefix}\n{piece}")
        content_hash = hash_text(normalize_for_hash(candidate))
        chunks.append({
            "source_hash": hash_text("|".join([vote_id, str(chunk_index), content_hash])),
            "vote_id": vote_id,
            "congress": safe_int(vote.get("congress")),
            "chamber": chamber,
            "session": text_value(vote.get("session")).strip(),
            "number": safe_int(vote.get("number")),
            "date": date,
            "category": text_value(vote.get("category")).strip(),
            "type": text_value(vote.get("type")).strip(),
            "question": text_value(vote.get("question")).strip(),
            "subject": text_value(vote.get("subject")).strip(),
            "result": text_value(vote.get("result")).strip(),
            "bill_id": bill_id or None,
            "text": candidate,
            "token_count": count_tokens(candidate),
            "chunk_index": chunk_index,
            "content_hash": content_hash,
        })
    return chunks


def write_congress_output(congress: int, votes: list[dict], output_root: Path, shard_size: int, max_tokens: int, overlap: int) -> dict:
    congress_dir = output_root / str(congress)
    congress_dir.mkdir(parents=True, exist_ok=True)

    for shard_path in congress_dir.glob("shard-*.jsonl"):
        shard_path.unlink()
    manifest_path = congress_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    chunk_records = []
    for vote in votes:
        chunk_records.extend(build_vote_chunks(vote, max_tokens=max_tokens, overlap=overlap))

    chunk_records.sort(key=lambda record: (
        text_value(record.get("date")),
        text_value(record.get("session")),
        text_value(record.get("chamber")),
        safe_int(record.get("number"), 0) or 0,
        text_value(record.get("vote_id")),
        safe_int(record.get("chunk_index"), 0) or 0,
    ))

    shards = shard_records(chunk_records, shard_size)
    manifest_shards = []
    total_tokens = 0
    for shard_index, shard in enumerate(shards, 1):
        shard_path = congress_dir / f"shard-{shard_index:05d}.jsonl"
        write_jsonl(shard_path, shard)
        shard_tokens = sum(safe_int(record.get("token_count"), 0) or 0 for record in shard)
        total_tokens += shard_tokens
        dates = [text_value(record.get("date")).strip() for record in shard if text_value(record.get("date")).strip()]
        manifest_shards.append({
            "shard": shard_path.name,
            "chunk_count": len(shard),
            "token_count": shard_tokens,
            "earliest_date": min(dates) if dates else None,
            "latest_date": max(dates) if dates else None,
        })

    dates = [text_value(vote.get("date")).strip() for vote in votes if text_value(vote.get("date")).strip()]
    manifest = {
        "congress": congress,
        "input_root": str(INPUT_ROOT),
        "output_root": str(output_root),
        "vote_count": len(votes),
        "chunk_count": len(chunk_records),
        "total_tokens": total_tokens,
        "shard_count": len(shards),
        "earliest_date": min(dates) if dates else None,
        "latest_date": max(dates) if dates else None,
        "shards": manifest_shards,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("[%s] wrote %s chunk(s) from %s vote(s) across %s shard(s)", congress, len(chunk_records), len(votes), len(shards))
    return manifest


def load_congress(congress: int, input_root: Path, output_root: Path, shard_size: int, max_tokens: int, overlap: int) -> dict | None:
    congress_dir = input_root / str(congress)
    if not congress_dir.is_dir():
        log.warning("[%s] no processed vote shards found", congress)
        return None

    votes = []
    for shard_path in sorted(congress_dir.glob("shard-*.jsonl")):
        votes.extend(read_jsonl(shard_path))

    votes.sort(key=lambda vote: (
        text_value(vote.get("date")),
        text_value(vote.get("session")),
        text_value(vote.get("chamber")),
        safe_int(vote.get("number"), 0) or 0,
        text_value(vote.get("vote_id")),
    ))
    return write_congress_output(congress, votes, output_root, shard_size, max_tokens, overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk normalized vote records into embedding-ready texts")
    parser.add_argument("--input-root", type=str, default=str(INPUT_ROOT),
                        help=f"Input vote shard root (default: {INPUT_ROOT})")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help=f"Output chunk root (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--congresses", type=int, nargs="*",
                        help="Congress numbers to process (default: all discovered)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help=f"Chunks per output shard (default: {DEFAULT_SHARD_SIZE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per chunk before splitting (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Token overlap for split chunks (default: {DEFAULT_OVERLAP})")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    congresses = args.congresses or discover_congresses(input_root)
    if not congresses:
        log.error("No processed vote input directories found under %s", input_root)
        return

    for congress in congresses:
        load_congress(congress, input_root, output_root, args.shard_size, args.max_tokens, args.overlap)


if __name__ == "__main__":
    main()
