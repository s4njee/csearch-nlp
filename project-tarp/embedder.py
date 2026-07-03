#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai",
#   "tiktoken",
# ]
# ///
"""
embedder.py — Convert text chunks into vector embeddings via OpenAI or Ollama.

Reads processed JSONL shard files, batches the text, calls the configured
embedding backend, and writes embedded JSONL shards that mirror the input
layout.

Supports incremental embedding: if an output shard already exists, only
missing/unembedded chunks in that shard are sent to the backend.

OpenAI:
    pip install openai
    export OPENAI_API_KEY=sk-...

Ollama:
    run a local or remote Ollama daemon with an embedding model such as
    qwen3-embedding:8b-q8_0

Usage:
    python embedder.py
    python embedder.py --backend openai --model text-embedding-3-small
    python embedder.py --backend ollama --model qwen3-embedding:8b-q8_0
    python embedder.py --batch-size 500
    python embedder.py --delay-seconds 3
    python embedder.py --checkpoint-every 10
    python embedder.py --dry-run
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "processed_chunks"
OUTPUT_FILE = DATA_DIR / "embedded_chunks"

DEFAULT_BACKEND = "openai"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_DIMENSIONS = None
DEFAULT_BATCH_SIZE = 500
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_BATCH_COMPLETION_WINDOW = "24h"
DEFAULT_OPENAI_BATCH_MAX_REQUESTS = 50000
DEFAULT_MAX_REQUEST_TOKENS = 250000

PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
BATCH_PRICING = {
    "text-embedding-3-small": 0.01,
    "text-embedding-3-large": 0.065,
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

def openai_embed_batch(client: OpenAI, texts: list[str], model: str, dimensions: Optional[int]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Returns list of vectors."""
    kwargs = {
        "input": texts,
        "model": model,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(
        **kwargs,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def ollama_embed_batch(base_url: str, texts: list[str], model: str) -> list[list[float]]:
    """Call Ollama /api/embed for a batch of texts. Returns list of vectors."""
    payload = json.dumps({
        "model": model,
        "input": texts,
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
    if not isinstance(embeddings, list):
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return embeddings


def embed_batch(client, backend: str, texts: list[str], model: str, dimensions: Optional[int], ollama_url: str) -> list[list[float]]:
    """Dispatch one embedding batch to the configured backend."""
    if backend == "openai":
        return openai_embed_batch(client, texts, model, dimensions)
    if backend == "ollama":
        return ollama_embed_batch(ollama_url, texts, model)
    raise ValueError(f"Unsupported backend: {backend}")


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
    source_hash = chunk.get("source_hash")
    if source_hash:
        return (str(source_hash), "", 0)
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


def chunk_token_count(chunk: dict) -> int:
    """Return the token count recorded for a chunk, or 0 if unavailable."""
    try:
        for key in ("token_count", "tokens"):
            value = chunk.get(key)
            if value not in (None, ""):
                return int(value)
        return 0
    except (TypeError, ValueError):
        return 0


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection for 429/rate-limit responses across client versions."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in text


def is_connection_error(exc: Exception) -> bool:
    """Best-effort detection for transient transport failures."""
    if isinstance(exc, urllib.error.URLError):
        return True

    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return any(
        token in name or token in text
        for token in (
            "connectionerror",
            "connection error",
            "apiconnectionerror",
            "timeout",
            "timed out",
            "readtimeout",
            "connecttimeout",
            "broken pipe",
            "connection reset",
            "remote disconnected",
            "temporarily unavailable",
        )
    )


def is_retryable_embedding_error(exc: Exception) -> bool:
    """Return True for transient embedding failures worth retrying."""
    return is_rate_limit_error(exc) or is_connection_error(exc)


def is_max_tokens_per_request_error(exc: Exception) -> bool:
    """Detect OpenAI request-size failures so we can split oversized batches."""
    text = str(exc).lower()
    return (
        "max_tokens_per_request" in text
        or "max tokens per request" in text
        or "requested" in text and "tokens" in text and "300000" in text
    )


def embed_texts_with_fallback(
    client,
    backend: str,
    texts: list[str],
    model: str,
    dimensions: Optional[int],
    ollama_url: str,
    max_split_depth: int = 12,
) -> list[list[float]]:
    """Embed a batch, splitting it on request-size failures if needed."""
    try:
        return embed_batch(client, backend, texts, model, dimensions, ollama_url)
    except Exception as exc:
        if backend != "openai" or not is_max_tokens_per_request_error(exc):
            raise
        if len(texts) <= 1 or max_split_depth <= 0:
            raise RuntimeError(
                f"OpenAI rejected a single-text embedding request as too large ({len(texts)} text(s))."
            ) from exc

        mid = max(1, len(texts) // 2)
        log.warning(
            f"OpenAI request exceeded token limit with {len(texts)} texts; "
            f"retrying as {mid} + {len(texts) - mid}"
        )
        left = embed_texts_with_fallback(
            client=client,
            backend=backend,
            texts=texts[:mid],
            model=model,
            dimensions=dimensions,
            ollama_url=ollama_url,
            max_split_depth=max_split_depth - 1,
        )
        right = embed_texts_with_fallback(
            client=client,
            backend=backend,
            texts=texts[mid:],
            model=model,
            dimensions=dimensions,
            ollama_url=ollama_url,
            max_split_depth=max_split_depth - 1,
        )
        return left + right


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


def batch_state_dir(output_root: Path) -> Path:
    return output_root / "_openai_batch"


def active_batch_state_path(output_root: Path) -> Path:
    return batch_state_dir(output_root) / "active_batch.json"


def batch_history_dir(output_root: Path) -> Path:
    return batch_state_dir(output_root) / "history"


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def chunk_identity_parts(chunk: dict) -> list:
    key = chunk_identity(chunk)
    return [key[0], key[1], key[2]]


def chunk_identity_from_parts(parts: list) -> tuple[str, str, int]:
    return (
        str(parts[0]) if len(parts) > 0 else "",
        str(parts[1]) if len(parts) > 1 else "",
        int(parts[2]) if len(parts) > 2 else 0,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value):
    """Best-effort conversion for SDK objects stored in local state files."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_batch_request_files(
    output_root: Path,
    model: str,
    dimensions: Optional[int],
    planned_chunks: list[dict],
) -> dict:
    """Create the next OpenAI batch input file and local request map."""
    base_dir = batch_state_dir(output_root)
    batch_dir = base_dir / f"job-{utc_now_compact()}"
    batch_dir.mkdir(parents=True, exist_ok=False)

    requests_path = batch_dir / "requests.jsonl"
    request_map_path = batch_dir / "request-map.jsonl"

    with requests_path.open("w", encoding="utf-8") as req_f, request_map_path.open("w", encoding="utf-8") as map_f:
        for idx, planned in enumerate(planned_chunks, 1):
            custom_id = f"emb-{idx:06d}"
            body = {
                "model": model,
                "input": planned["text"],
            }
            if dimensions is not None:
                body["dimensions"] = dimensions

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": body,
            }
            req_f.write(json.dumps(request, separators=(",", ":")))
            req_f.write("\n")

            mapping = {
                "custom_id": custom_id,
                "input_shard": planned["input_shard"],
                "identity": chunk_identity_parts(planned),
            }
            map_f.write(json.dumps(mapping, separators=(",", ":")))
            map_f.write("\n")

    return {
        "batch_dir": str(batch_dir),
        "requests_path": str(requests_path),
        "request_map_path": str(request_map_path),
        "request_count": len(planned_chunks),
    }


def read_batch_request_map(path: Path) -> dict[str, dict]:
    mapping = {}
    for record in read_jsonl(path):
        mapping[record["custom_id"]] = record
    return mapping


def download_openai_file(client: OpenAI, file_id: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    data = content.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    destination.write_bytes(data)
    return destination


def openai_batch_terminal(status: str) -> bool:
    return status in {"completed", "failed", "expired", "cancelled"}


def is_enqueued_limit_failure(batch) -> bool:
    """Check if a failed batch was rejected due to the enqueued token limit."""
    errors = getattr(batch, "errors", None)
    if not errors:
        return False
    for err in getattr(errors, "data", None) or []:
        code = getattr(err, "code", "")
        message = getattr(err, "message", "").lower()
        if code == "token_limit_exceeded" or "enqueued" in message:
            return True
    return False


def merge_openai_batch_results(output_root: Path, batch_state: dict) -> tuple[int, int]:
    """Merge a completed OpenAI batch output file into mirrored shard outputs."""
    batch_dir = Path(batch_state["batch_dir"])
    request_map = read_batch_request_map(Path(batch_state["request_map_path"]))
    output_file = batch_dir / "output.jsonl"
    if not output_file.exists():
        raise RuntimeError(f"Batch output file not found: {output_file}")

    vectors_by_shard: dict[str, dict[tuple[str, str, int], list[float]]] = {}
    success_count = 0
    failed_count = 0

    for record in read_jsonl(output_file):
        custom_id = record.get("custom_id")
        if not custom_id or custom_id not in request_map:
            continue

        response = record.get("response", {})
        status_code = int(response.get("status_code", 0) or 0)
        if status_code != 200:
            failed_count += 1
            continue

        body = response.get("body", {})
        data = body.get("data", [])
        if not data:
            failed_count += 1
            continue

        embedding = data[0].get("embedding")
        if embedding is None:
            failed_count += 1
            continue

        mapping = request_map[custom_id]
        shard_rel = mapping["input_shard"]
        shard_vectors = vectors_by_shard.setdefault(shard_rel, {})
        shard_vectors[chunk_identity_from_parts(mapping["identity"])] = embedding
        success_count += 1

    for shard_rel, shard_vectors in vectors_by_shard.items():
        input_shard = Path(batch_state["input_root"]) / shard_rel
        output_shard = Path(batch_state["output_root"]) / shard_rel
        input_chunks = read_jsonl(input_shard)
        embedded_by_key = load_existing_shard(output_shard)

        for chunk in input_chunks:
            key = chunk_identity(chunk)
            vector = shard_vectors.get(key)
            if vector is None:
                continue
            updated = dict(chunk)
            updated["embedding"] = vector
            embedded_by_key[key] = updated

        write_jsonl(output_shard, build_shard_records(input_chunks, embedded_by_key))

    return success_count, failed_count


def poll_openai_batch(client: OpenAI, batch_id: str, poll_interval: float) -> object:
    """Poll an OpenAI batch until it reaches a terminal state."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if openai_batch_terminal(batch.status):
            return batch
        log.info(f"Batch {batch_id} status: {batch.status}; sleeping {poll_interval:.1f}s")
        time.sleep(poll_interval)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completed_shard_manifest(
    output_root: Path,
    expected_model: Optional[str] = None,
    expected_dimensions: Optional[int] = None,
) -> dict[str, dict]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = load_json(manifest_path)
    except (json.JSONDecodeError, OSError):
        return {}

    if expected_model is not None and manifest.get("model") != expected_model:
        return {}
    if expected_dimensions is not None and manifest.get("dimensions") != expected_dimensions:
        return {}

    completed = {}
    for status in manifest.get("shards", []):
        if status.get("to_embed") != 0:
            continue
        shard_key = status.get("input_shard_rel") or status.get("input_shard")
        if shard_key and status.get("input_sha256"):
            completed[str(shard_key)] = status
    return completed


def collect_pending_chunks(
    input_path: Path,
    output_path: Path,
    shard_paths: list[Path],
    expected_model: Optional[str] = None,
    expected_dimensions: Optional[int] = None,
) -> tuple[list[dict], int, int, int, list[dict]]:
    """Return pending chunks with shard context plus aggregate counts."""
    pending = []
    total_chunks = 0
    total_tokens = 0
    total_already_done = 0
    shard_statuses = []
    completed_manifest = load_completed_shard_manifest(
        output_path,
        expected_model=expected_model,
        expected_dimensions=expected_dimensions,
    )

    for shard_path in shard_paths:
        output_shard = shard_output_path(input_path, output_path, shard_path)
        shard_rel = str(shard_path.relative_to(input_path)) if input_path.is_dir() else shard_path.name
        input_sha256 = sha256_file(shard_path)
        previous_status = completed_manifest.get(shard_rel) or completed_manifest.get(str(shard_path))
        if (
            previous_status
            and previous_status.get("input_sha256") == input_sha256
            and output_shard.exists()
        ):
            shard_total = int(previous_status.get("chunk_count", 0))
            total_chunks += shard_total
            total_already_done += shard_total
            shard_statuses.append({
                "input_shard": str(shard_path),
                "input_shard_rel": shard_rel,
                "output_shard": str(output_shard),
                "input_sha256": input_sha256,
                "chunk_count": shard_total,
                "already_embedded": shard_total,
                "to_embed": 0,
                "skipped_by_manifest": True,
            })
            continue

        input_chunks = read_jsonl(shard_path)
        existing_by_key = load_existing_shard(output_shard)
        shard_total = len(input_chunks)
        shard_pending = [chunk for chunk in input_chunks if chunk_identity(chunk) not in existing_by_key]
        shard_pending_tokens = sum(chunk_token_count(chunk) for chunk in shard_pending)

        total_chunks += shard_total
        total_tokens += shard_pending_tokens
        total_already_done += shard_total - len(shard_pending)
        shard_statuses.append({
            "input_shard": str(shard_path),
            "input_shard_rel": shard_rel,
            "output_shard": str(output_shard),
            "input_sha256": input_sha256,
            "chunk_count": shard_total,
            "already_embedded": shard_total - len(shard_pending),
            "to_embed": len(shard_pending),
        })

        for chunk in shard_pending:
            pending.append({
                **chunk,
                "input_shard": shard_rel,
            })

    return pending, total_chunks, total_tokens, total_already_done, shard_statuses


def pack_chunks_for_embedding(chunks: list[dict], max_chunks: int, max_tokens: int) -> list[list[dict]]:
    """
    Pack chunks into request-sized batches while staying under both chunk-count
    and token-count limits.
    """
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for chunk in chunks:
        tokens = chunk_token_count(chunk)
        if tokens > max_tokens:
            identity = chunk_identity(chunk)
            raise ValueError(
                f"Chunk {identity} has {tokens} tokens, which exceeds the per-request limit of {max_tokens}."
            )

        would_exceed_count = len(current) >= max_chunks
        would_exceed_tokens = current and (current_tokens + tokens > max_tokens)
        if current and (would_exceed_count or would_exceed_tokens):
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(chunk)
        current_tokens += tokens

    if current:
        batches.append(current)

    return batches


def run_openai_batch_backend(client: OpenAI, args, input_path: Path, output_path: Path, shard_paths: list[Path], shard_statuses: list[dict], pending_chunks: list[dict], total_embedded_now: int) -> int:
    """Submit or collect one active OpenAI batch job and merge results into shard outputs."""
    state_path = active_batch_state_path(output_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        batch_state = load_json(state_path)
        batch = client.batches.retrieve(batch_state["batch_id"])
        batch_state["status"] = batch.status
        batch_state["request_counts"] = to_jsonable(getattr(batch, "request_counts", None))
        save_json(state_path, batch_state)
        log.info(f"Active batch {batch.id} status: {batch.status}")

        if not openai_batch_terminal(batch.status):
            if args.wait_for_batch:
                batch = poll_openai_batch(client, batch.id, args.poll_seconds)
                batch_state["status"] = batch.status
                batch_state["request_counts"] = to_jsonable(getattr(batch, "request_counts", None))
                save_json(state_path, batch_state)
            else:
                log.info("Batch is still running. Re-run later or use --wait-for-batch.")
                return total_embedded_now

        batch_dir = Path(batch_state["batch_dir"])
        if getattr(batch, "output_file_id", None):
            download_openai_file(client, batch.output_file_id, batch_dir / "output.jsonl")
        if getattr(batch, "error_file_id", None):
            download_openai_file(client, batch.error_file_id, batch_dir / "errors.jsonl")

        batch_state["status"] = batch.status
        batch_state["output_file_id"] = getattr(batch, "output_file_id", None)
        batch_state["error_file_id"] = getattr(batch, "error_file_id", None)
        batch_state["completed_at"] = getattr(batch, "completed_at", None)
        batch_state["failed_at"] = getattr(batch, "failed_at", None)
        batch_state["expired_at"] = getattr(batch, "expired_at", None)
        batch_state["cancelled_at"] = getattr(batch, "cancelled_at", None)

        if batch.status == "completed" and (batch_dir / "output.jsonl").exists():
            success_count, failed_count = merge_openai_batch_results(output_path, batch_state)
            batch_state["merged_embeddings"] = success_count
            batch_state["failed_embeddings"] = failed_count
            total_embedded_now += success_count
            log.info(
                f"Merged batch {batch.id}: {success_count} embeddings applied, {failed_count} failed responses"
            )
        else:
            log.warning(f"Batch {batch.id} ended with status {batch.status}; no embeddings were merged")
            errors = getattr(batch, "errors", None)
            if errors:
                error_data = getattr(errors, "data", None) or []
                for err in error_data[:5]:
                    code = getattr(err, "code", "?")
                    message = getattr(err, "message", "?")
                    log.error(f"  Batch error: [{code}] {message}")
            errors_file = batch_dir / "errors.jsonl"
            if errors_file.exists():
                log.info(f"  Full error details: {errors_file}")

            # Retry if the batch was rejected due to enqueued token limits
            if is_enqueued_limit_failure(batch) and args.wait_for_batch:
                enqueue_retries = getattr(args, "_enqueue_retries", 0)
                if enqueue_retries < args.max_retries:
                    args._enqueue_retries = enqueue_retries + 1
                    backoff = args.poll_seconds * (2 ** enqueue_retries)
                    log.info(
                        f"Enqueued token limit hit (attempt {enqueue_retries + 1}/{args.max_retries}); "
                        f"waiting {backoff:.0f}s before resubmitting..."
                    )
                    history_path = batch_history_dir(output_path) / f"{Path(batch_state['batch_dir']).name}.json"
                    save_json(history_path, batch_state)
                    state_path.unlink()
                    time.sleep(backoff)
                    # Fall through to resubmit below
                    pending_chunks, _, _, _, shard_statuses[:] = collect_pending_chunks(
                        input_path,
                        output_path,
                        shard_paths,
                        expected_model=args.model,
                        expected_dimensions=args.dimensions,
                    )
                else:
                    log.error(f"Gave up after {args.max_retries} enqueued-limit retries")
                    history_path = batch_history_dir(output_path) / f"{Path(batch_state['batch_dir']).name}.json"
                    save_json(history_path, batch_state)
                    state_path.unlink()
                    _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
                    return total_embedded_now
            else:
                history_path = batch_history_dir(output_path) / f"{Path(batch_state['batch_dir']).name}.json"
                save_json(history_path, batch_state)
                state_path.unlink()

                pending_chunks, _, _, _, shard_statuses[:] = collect_pending_chunks(
                    input_path,
                    output_path,
                    shard_paths,
                    expected_model=args.model,
                    expected_dimensions=args.dimensions,
                )
                if not pending_chunks:
                    _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
                    return total_embedded_now

    if not pending_chunks:
        log.info("All chunks are already embedded.")
        _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
        return total_embedded_now

    try:
        planned_batches = pack_chunks_for_embedding(
            pending_chunks,
            max_chunks=args.batch_api_max_requests,
            max_tokens=args.max_request_tokens,
        )
    except ValueError as e:
        log.error(str(e))
        return total_embedded_now
    planned_chunks = planned_batches[0]
    batch_files = write_batch_request_files(output_path, args.model, args.dimensions, planned_chunks)
    requests_path = Path(batch_files["requests_path"])

    with requests_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    for attempt in range(args.max_retries + 1):
        try:
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/embeddings",
                completion_window=args.batch_api_completion_window,
                metadata={
                    "project": "project-tarp",
                    "model": args.model,
                },
            )
            break
        except Exception as e:
            msg = str(e).lower()
            is_enqueued_limit = "enqueued" in msg and "limit" in msg
            if (is_enqueued_limit or is_rate_limit_error(e)) and attempt < args.max_retries:
                backoff = args.poll_seconds * (2 ** attempt)
                log.warning(
                    f"Batch submission blocked (attempt {attempt + 1}/{args.max_retries + 1}): {e}"
                )
                log.info(f"Waiting {backoff:.0f}s for in-progress batches to drain...")
                time.sleep(backoff)
                continue
            raise

    state = {
        "backend": args.backend,
        "model": args.model,
        "dimensions": args.dimensions,
        "input_root": str(input_path),
        "output_root": str(output_path),
        "batch_dir": batch_files["batch_dir"],
        "requests_path": batch_files["requests_path"],
        "request_map_path": batch_files["request_map_path"],
        "request_count": batch_files["request_count"],
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "completion_window": args.batch_api_completion_window,
        "created_at": getattr(batch, "created_at", None),
    }
    save_json(state_path, state)
    log.info(
        f"Submitted OpenAI batch {batch.id} with {batch_files['request_count']} embedding requests "
        f"from {requests_path}"
    )

    if args.wait_for_batch:
        return run_openai_batch_backend(
            client=client,
            args=args,
            input_path=input_path,
            output_path=output_path,
            shard_paths=shard_paths,
            shard_statuses=shard_statuses,
            pending_chunks=planned_chunks,
            total_embedded_now=total_embedded_now,
        )

    _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
    return total_embedded_now


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Embed text chunks via OpenAI or Ollama")
    parser.add_argument("--input", "--input-dir", dest="input", type=str, default=str(INPUT_FILE),
                        help=f"Input chunk shard directory (default: {INPUT_FILE})")
    parser.add_argument("--output", "--output-dir", dest="output", type=str, default=str(OUTPUT_FILE),
                        help=f"Output embedded shard directory (default: {OUTPUT_FILE})")
    parser.add_argument("--backend", choices=["openai", "openai-batch", "ollama"], default=os.environ.get("EMBEDDING_BACKEND", DEFAULT_BACKEND),
                        help=f"Embedding backend (default: {os.environ.get('EMBEDDING_BACKEND', DEFAULT_BACKEND)})")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedding model name; defaults depend on backend")
    parser.add_argument("--dimensions", type=int, default=None,
                        help="Requested embedding dimensions for backends that support it")
    parser.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
                        help=f"Ollama base URL (default: {os.environ.get('OLLAMA_HOST', DEFAULT_OLLAMA_URL)})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Texts per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS,
                        help=f"Seconds to wait between successful batches (default: {DEFAULT_DELAY_SECONDS})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"Max retries for rate-limited batches (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help=f"Save checkpoint every N successful batches within a shard (default: {DEFAULT_CHECKPOINT_EVERY})")
    parser.add_argument("--max-request-tokens", type=int, default=DEFAULT_MAX_REQUEST_TOKENS,
                        help=f"Max tokens per embedding request (default: {DEFAULT_MAX_REQUEST_TOKENS})")
    parser.add_argument("--batch-api-max-requests", type=int, default=DEFAULT_OPENAI_BATCH_MAX_REQUESTS,
                        help=f"Max embedding requests per OpenAI Batch job (default: {DEFAULT_OPENAI_BATCH_MAX_REQUESTS})")
    parser.add_argument("--batch-api-completion-window", type=str, default=DEFAULT_OPENAI_BATCH_COMPLETION_WINDOW,
                        help=f"OpenAI Batch completion window (default: {DEFAULT_OPENAI_BATCH_COMPLETION_WINDOW})")
    parser.add_argument("--wait-for-batch", action="store_true",
                        help="For openai-batch, poll until the active batch reaches a terminal state and collect it")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="Polling interval in seconds for --wait-for-batch (default: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without calling the API")
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODEL if args.backend in {"openai", "openai-batch"} else DEFAULT_OLLAMA_MODEL
    if args.dimensions is None and args.backend in {"openai", "openai-batch"}:
        args.dimensions = 1536 if args.model == "text-embedding-3-small" else None

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error(f"Input path not found: {input_path}")
        log.error("Run chunker.py first to generate processed chunk shards")
        return 1

    try:
        shard_paths = iter_input_shards(input_path)
    except ValueError as e:
        log.error(str(e))
        return 1

    if not shard_paths:
        log.error(f"No input shard files found under {input_path}")
        return 1

    pending_chunks, total_chunks, total_tokens, total_already_done, shard_statuses = collect_pending_chunks(
        input_path,
        output_path,
        shard_paths,
        expected_model=args.model,
        expected_dimensions=args.dimensions,
    )

    if total_chunks == 0:
        log.info("No chunks found. Nothing to do.")
        return 0

    log.info(f"Discovered {len(shard_paths)} shard(s) with {total_chunks} chunk(s) total")
    log.info(f"Need to embed: {total_chunks - total_already_done} chunks ({total_already_done} already done)")

    if args.backend in {"openai", "openai-batch"}:
        pricing = BATCH_PRICING if args.backend == "openai-batch" else PRICING
        price_per_m = pricing.get(args.model, PRICING.get(args.model, 0.02))
        est_cost = total_tokens / 1_000_000 * price_per_m
        log.info(f"Estimated cost: {total_tokens:,} tokens × ${price_per_m}/1M = ${est_cost:.4f}")
    else:
        log.info(f"Backend: ollama | host={args.ollama_url} | estimated tokens to embed: {total_tokens:,}")

    if args.dry_run:
        log.info(f"[DRY RUN] Exiting without calling {args.backend}")
        return 0

    client = None
    if args.backend in {"openai", "openai-batch"}:
        if OpenAI is None:
            log.error("openai package not installed. Run: pip install openai")
            return 1
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OPENAI_API_KEY environment variable not set")
            return 1
        client = OpenAI(api_key=api_key)
    else:
        if args.dimensions is not None:
            log.warning("--dimensions is ignored for Ollama /api/embed; use the model's native output size")

    total_embedded_now = 0
    if args.backend == "openai-batch":
        total_embedded_now = run_openai_batch_backend(
            client=client,
            args=args,
            input_path=input_path,
            output_path=output_path,
            shard_paths=shard_paths,
            shard_statuses=shard_statuses,
            pending_chunks=pending_chunks,
            total_embedded_now=total_embedded_now,
        )
        log.info(f"{'=' * 60}")
        log.info(f"Batch mode progress applied in this run: {total_embedded_now} embeddings merged")
        log.info(f"Batch state directory: {batch_state_dir(output_path)}")
        remaining_chunks, _, _, _, _ = collect_pending_chunks(
            input_path,
            output_path,
            shard_paths,
            expected_model=args.model,
            expected_dimensions=args.dimensions,
        )
        if remaining_chunks:
            log.error(
                "OpenAI batch mode still has %s pending chunk(s); refusing to report success before all embeddings are merged",
                len(remaining_chunks),
            )
            return 1
        return 0

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

            try:
                batches = pack_chunks_for_embedding(
                    to_embed,
                    max_chunks=args.batch_size,
                    max_tokens=args.max_request_tokens,
                )
            except ValueError as e:
                log.error(str(e))
                _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
                return 1
            log.info(
                f"Shard {shard_index}/{len(shard_paths)}: {shard_path.name} | "
                f"{len(to_embed)}/{len(input_chunks)} chunks to embed | "
                f"{len(batches)} batch(es)"
            )
            log.info(
                f"Inter-batch delay: {args.delay_seconds:.1f}s | "
                f"Max rate-limit retries: {args.max_retries} | "
                f"Checkpoint every {args.checkpoint_every} batch(es) | "
                f"Max request tokens: {args.max_request_tokens:,}"
            )

            shard_embedded_now = 0
            for bi, batch in enumerate(batches, 1):
                texts = [chunk["text"] for chunk in batch]

                for attempt in range(args.max_retries + 1):
                    try:
                        vectors = embed_texts_with_fallback(
                            client=client,
                            backend=args.backend,
                            texts=texts,
                            model=args.model,
                            dimensions=args.dimensions,
                            ollama_url=args.ollama_url,
                        )
                        break
                    except Exception as e:
                        if is_retryable_embedding_error(e) and attempt < args.max_retries:
                            backoff = args.delay_seconds * (2 ** attempt)
                            reason = "rate limited" if is_rate_limit_error(e) else "connection failed"
                            log.warning(
                                f"Transient embedding error ({reason}) on shard {shard_index} batch {bi} "
                                f"(attempt {attempt + 1}/{args.max_retries + 1}); "
                                f"sleeping {backoff:.1f}s before retry"
                            )
                            time.sleep(backoff)
                            continue

                        if is_max_tokens_per_request_error(e):
                            log.error(
                                f"API error on shard {shard_index} batch {bi}: {e}\n"
                                f"Try lowering --max-request-tokens or --batch-size if this keeps happening."
                            )
                        else:
                            log.error(f"API error on shard {shard_index} batch {bi}: {e}")
                        write_jsonl(output_shard, build_shard_records(input_chunks, embedded_by_key))
                        _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
                        return 1

                for chunk, vector in zip(batch, vectors):
                    updated = dict(chunk)
                    updated["embedding"] = vector
                    embedded_by_key[chunk_identity(updated)] = updated
                    shard_embedded_now += 1
                    total_embedded_now += 1

                batch_tokens = sum(chunk_token_count(chunk) for chunk in batch)
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
        return 130

    _save_final_manifest(output_path, args, shard_statuses, total_embedded_now)
    log.info(f"{'=' * 60}")
    log.info(f"DONE: {total_embedded_now} chunks embedded in this run")
    log.info(f"Saved embedded shards under {output_path}")
    return 0


def _save_final_manifest(output_root: Path, args, shard_statuses: list[dict], embedded_now: int) -> None:
    """Save a lightweight manifest describing the embedded shard directory."""
    summary = {
        "backend": args.backend,
        "model": args.model,
        "dimensions": args.dimensions,
        "ollama_url": args.ollama_url if args.backend == "ollama" else "",
        "batch_size": args.batch_size,
        "delay_seconds": args.delay_seconds,
        "max_retries": args.max_retries,
        "checkpoint_every": args.checkpoint_every,
        "max_request_tokens": args.max_request_tokens,
        "embedded_in_this_run": embedded_now,
        "shards": shard_statuses,
    }
    save_manifest(output_root, summary)


if __name__ == "__main__":
    sys.exit(main())
