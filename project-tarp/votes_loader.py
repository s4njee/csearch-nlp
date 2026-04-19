#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
from __future__ import annotations
"""
votes_loader.py — Walk the scraper's vote tree and emit normalized vote records.

Reads @unitedstates/congress vote JSON from:
    backend/scraper/congress/data/{congress}/votes/{session}/{chamber}{number}/data.json

and writes rolling JSONL shards under:
    backend/nlp/project-tarp/data/processed_votes/{congress}/

The loader keeps only vote-level metadata. Per-legislator vote arrays are
intentionally excluded so they do not bloat the embedding pipeline.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRAPER_DATA_ROOT = REPO_ROOT / "backend" / "scraper" / "congress" / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "processed_votes"
DEFAULT_SHARD_SIZE = 2000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("votes_loader")


def safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def text_value(value) -> str:
    return "" if value is None else str(value)


def normalize_bill_id(bill: dict | None) -> str | None:
    if not isinstance(bill, dict):
        return None

    bill_type = text_value(bill.get("type")).strip()
    bill_number = bill.get("number")
    bill_congress = safe_int(bill.get("congress"))
    if not bill_type or bill_number in (None, "") or bill_congress is None:
        return None
    return f"{bill_type}{bill_number}-{bill_congress}"


def discover_congresses(data_root: Path) -> list[int]:
    if not data_root.exists():
        return []
    congresses = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and path.name.isdigit() and (path / "votes").is_dir():
            congresses.append(int(path.name))
    return congresses


def iter_vote_json_files(data_root: Path, congress: int) -> list[Path]:
    votes_root = data_root / str(congress) / "votes"
    if not votes_root.is_dir():
        return []

    files = []
    for session_dir in sorted(votes_root.iterdir()):
        if not session_dir.is_dir():
            continue
        for chamber_dir in sorted(session_dir.iterdir()):
            if not chamber_dir.is_dir():
                continue
            data_json = chamber_dir / "data.json"
            if data_json.is_file():
                files.append(data_json)
    return files


def load_vote(path: Path, repo_root: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("Skipping malformed vote JSON %s: %s", path, exc)
        return None

    question = text_value(raw.get("question")).strip()
    subject = text_value(raw.get("subject")).strip()
    if not question and not subject:
        log.warning("Skipping vote with empty question and subject: %s", path)
        return None

    vote_id = text_value(raw.get("vote_id")).strip()
    if not vote_id:
        log.warning("Skipping vote with missing vote_id: %s", path)
        return None

    source_path = path
    try:
        source_path = path.relative_to(repo_root)
    except ValueError:
        pass

    return {
        "vote_id": vote_id,
        "congress": safe_int(raw.get("congress")),
        "session": text_value(raw.get("session")).strip(),
        "chamber": text_value(raw.get("chamber")).strip().lower(),
        "number": safe_int(raw.get("number")),
        "date": text_value(raw.get("date")).strip(),
        "category": text_value(raw.get("category")).strip(),
        "type": text_value(raw.get("type")).strip(),
        "question": question,
        "subject": subject,
        "result": text_value(raw.get("result")).strip(),
        "requires": text_value(raw.get("requires")).strip(),
        "source_url": text_value(raw.get("source_url")).strip(),
        "bill_id": normalize_bill_id(raw.get("bill")),
        "source_path": source_path.as_posix(),
    }


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


def date_bounds(records: list[dict]) -> tuple[str | None, str | None]:
    dates = [text_value(record.get("date")).strip() for record in records if text_value(record.get("date")).strip()]
    if not dates:
        return None, None
    return min(dates), max(dates)


def write_congress_output(congress: int, votes: list[dict], output_root: Path, shard_size: int, data_root: Path) -> dict:
    congress_dir = output_root / str(congress)
    congress_dir.mkdir(parents=True, exist_ok=True)

    for shard_path in congress_dir.glob("shard-*.jsonl"):
        shard_path.unlink()
    manifest_path = congress_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    shards = shard_records(votes, shard_size)
    manifest_shards = []
    for shard_index, shard in enumerate(shards, 1):
        shard_path = congress_dir / f"shard-{shard_index:05d}.jsonl"
        write_jsonl(shard_path, shard)
        earliest, latest = date_bounds(shard)
        manifest_shards.append({
            "shard": shard_path.name,
            "vote_count": len(shard),
            "earliest_date": earliest,
            "latest_date": latest,
        })

    earliest, latest = date_bounds(votes)
    manifest = {
        "congress": congress,
        "input_root": str(data_root),
        "output_root": str(output_root),
        "vote_count": len(votes),
        "shard_count": len(shards),
        "earliest_date": earliest,
        "latest_date": latest,
        "shards": manifest_shards,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("[%s] wrote %s vote record(s) across %s shard(s)", congress, len(votes), len(shards))
    return manifest


def load_congress(congress: int, data_root: Path, output_root: Path, shard_size: int) -> dict | None:
    vote_files = iter_vote_json_files(data_root, congress)
    if not vote_files:
        log.warning("[%s] no vote files found", congress)
        return None

    votes = []
    for path in vote_files:
        vote = load_vote(path, REPO_ROOT)
        if vote is not None:
            votes.append(vote)

    votes.sort(key=lambda vote: (
        text_value(vote.get("date")),
        text_value(vote.get("session")),
        text_value(vote.get("chamber")),
        safe_int(vote.get("number"), 0) or 0,
        text_value(vote.get("vote_id")),
    ))
    return write_congress_output(congress, votes, output_root, shard_size, data_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Congressional vote JSON into shardable records")
    parser.add_argument("--data-root", type=str, default=str(SCRAPER_DATA_ROOT),
                        help=f"Scraper data root (default: {SCRAPER_DATA_ROOT})")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--congresses", type=int, nargs="*",
                        help="Congress numbers to process (default: all discovered)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help=f"Votes per shard (default: {DEFAULT_SHARD_SIZE})")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    congresses = args.congresses or discover_congresses(data_root)
    if not congresses:
        log.error("No congress vote directories found under %s", data_root)
        return

    log.info("Processing %s congress(es): %s", len(congresses), ", ".join(str(c) for c in congresses))
    for congress in congresses:
        load_congress(congress, data_root, output_root, args.shard_size)


if __name__ == "__main__":
    main()
