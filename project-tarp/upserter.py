#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg2-binary",
# ]
# ///
"""
upserter.py — Load embedded chunk shards into PostgreSQL with pgvector.

Reads embedded JSONL shard files produced by embedder.py, stages each shard in
PostgreSQL, and bulk loads rows into:

  - nlp.bill_chunks
  - nlp.bill_embeddings

The loader keeps shard ingestion idempotent by deleting any existing rows for
the shard's bill_ids before inserting fresh chunk and embedding rows.

The HNSW index is built in one bulk pass after all shards are loaded for
maximum build quality and speed.

Usage:
    uv run upserter.py                    # load shards (DB must already exist)
    uv run upserter.py --create-db        # create DB + extension if needed
    uv run upserter.py --dry-run          # validate shards without loading
    uv run upserter.py --recreate         # drop and recreate tables before loading
    uv run upserter.py --dsn postgresql://user:pass@host:5432/csearch
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2.extras import Json, execute_values


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_DIR = DATA_DIR / "embedded_chunks"

DEFAULT_DSN = os.environ.get("PG_CONNECTION_STRING", "")
DEFAULT_SCHEMA = "nlp"
DEFAULT_CHUNK_TABLE = "bill_chunks"
DEFAULT_EMBEDDING_TABLE = "bill_embeddings"
DEFAULT_VOTE_CHUNK_TABLE = "vote_chunks"
DEFAULT_VOTE_EMBEDDING_TABLE = "vote_embeddings"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_VECTOR_SIZE = 1536
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 128
DEFAULT_MAINTENANCE_WORK_MEM = "16GB"
DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS = 6
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


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def text_value(value) -> str:
    return "" if value is None else str(value)


def parse_timestamptz(value):
    if value in (None, ""):
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def chunk_source_hash(chunk: dict) -> str:
    raw = "|".join([
        str(chunk.get("bill_id", "")),
        str(chunk.get("document_text_hash", "")),
        str(chunk.get("section_text_hash", "")),
        str(chunk.get("chunk_index", 0)),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vote_source_hash(chunk: dict) -> str:
    source_hash = text_value(chunk.get("source_hash")).strip()
    if source_hash:
        return source_hash
    raw = "|".join([
        text_value(chunk.get("vote_id")),
        text_value(chunk.get("chunk_index", 0)),
        text_value(chunk.get("content_hash")),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chunk_section_path(chunk: dict) -> Optional[str]:
    section_enum = (chunk.get("section_enum") or "").strip()
    section_header = (chunk.get("section_header") or "").strip()
    path = " ".join(part for part in [section_enum, section_header] if part)
    return path or None


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.17g}" for v in values) + "]"


def build_chunk_row(chunk: dict, model: str, include_aliases: bool) -> tuple:
    source_hash = chunk_source_hash(chunk)
    return (
        source_hash,
        text_value(chunk.get("bill_id")),
        text_value(chunk.get("canonical_bill_id", chunk.get("bill_id"))),
        safe_int(chunk.get("congress")),
        text_value(chunk.get("type")),
        text_value(chunk.get("number")),
        "section",
        chunk_section_path(chunk),
        text_value(chunk.get("short_title")),
        text_value(chunk.get("text")),
        safe_int(chunk.get("tokens")),
        text_value(chunk.get("version")),
        text_value(chunk.get("status")),
        text_value(chunk.get("section_enum")),
        text_value(chunk.get("section_header")),
        safe_int(chunk.get("chunk_index")),
        safe_int(chunk.get("original_chunk_index"), safe_int(chunk.get("chunk_index"))),
        text_value(chunk.get("document_text_hash")),
        text_value(chunk.get("section_text_hash")),
        Json(chunk.get("document_text_hashes", [])),
        Json(chunk.get("document_aliases", [])) if include_aliases else None,
        Json(chunk.get("section_aliases", [])) if include_aliases else None,
        vector_literal(chunk["embedding"]),
        model,
    )


def build_vote_row(chunk: dict, model: str) -> tuple:
    source_hash = vote_source_hash(chunk)
    body = text_value(chunk.get("text", chunk.get("body", "")))
    return (
        source_hash,
        text_value(chunk.get("vote_id")),
        safe_int(chunk.get("congress")),
        text_value(chunk.get("chamber")),
        text_value(chunk.get("session")),
        safe_int(chunk.get("number")),
        parse_timestamptz(chunk.get("date")),
        text_value(chunk.get("category")),
        text_value(chunk.get("type")),
        text_value(chunk.get("question")),
        text_value(chunk.get("subject")),
        text_value(chunk.get("result")),
        text_value(chunk.get("bill_id")) or None,
        body,
        safe_int(chunk.get("token_count", chunk.get("tokens"))),
        safe_int(chunk.get("chunk_index")),
        text_value(chunk.get("content_hash")),
        vector_literal(chunk["embedding"]),
        model,
    )


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def dsn_for_maintenance(dsn: str) -> str:
    """Swap the database in a DSN to 'postgres' for CREATE DATABASE."""
    parsed = urlparse(dsn)
    maintenance = parsed._replace(path="/postgres")
    return urlunparse(maintenance)


def create_database_if_not_exists(dsn: str) -> None:
    """Create the target database and enable pgvector if they don't exist."""
    parsed = urlparse(dsn)
    db_name = parsed.path.lstrip("/")
    maintenance_dsn = dsn_for_maintenance(dsn)

    conn = psycopg2.connect(maintenance_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                log.info(f"Database '{db_name}' already exists — skipping creation")
            else:
                cur.execute(f'CREATE DATABASE "{db_name}"')
                log.info(f"Created database '{db_name}'")
    finally:
        conn.close()

    # Enable pgvector in the target database
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            log.info("pgvector extension enabled")
    finally:
        conn.close()


def ensure_schema(conn, schema: str, chunk_table: str, embedding_table: str, vector_size: int, recreate: bool) -> None:
    """Create the schema and tables. HNSW index is built separately after load."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        if recreate:
            log.info("Dropping existing tables before recreate")
            cur.execute(f"DROP TABLE IF EXISTS {schema}.{embedding_table} CASCADE")
            cur.execute(f"DROP TABLE IF EXISTS {schema}.{chunk_table} CASCADE")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{chunk_table} (
              id BIGSERIAL PRIMARY KEY,
              source_hash TEXT NOT NULL UNIQUE,
              bill_id TEXT NOT NULL,
              canonical_bill_id TEXT NOT NULL,
              congress INTEGER NOT NULL,
              bill_type TEXT NOT NULL,
              bill_number TEXT NOT NULL,
              chunk_type TEXT NOT NULL,
              section_path TEXT,
              title TEXT,
              body TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              source_version TEXT,
              status TEXT,
              section_enum TEXT,
              section_header TEXT,
              chunk_index INTEGER NOT NULL,
              original_chunk_index INTEGER NOT NULL,
              document_text_hash TEXT NOT NULL,
              section_text_hash TEXT NOT NULL,
              document_text_hashes JSONB NOT NULL DEFAULT '[]'::jsonb,
              document_aliases JSONB,
              section_aliases JSONB,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{embedding_table} (
              chunk_id BIGINT PRIMARY KEY REFERENCES {schema}.{chunk_table}(id) ON DELETE CASCADE,
              embedding vector({vector_size}) NOT NULL,
              model TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        for sql in [
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_bill_id_idx ON {schema}.{chunk_table} (bill_id)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_canonical_bill_id_idx ON {schema}.{chunk_table} (canonical_bill_id)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_congress_idx ON {schema}.{chunk_table} (congress)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_bill_type_idx ON {schema}.{chunk_table} (bill_type)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_chunk_type_idx ON {schema}.{chunk_table} (chunk_type)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_source_hash_idx ON {schema}.{chunk_table} (source_hash)",
        ]:
            cur.execute(sql)


def ensure_vote_schema(conn, schema: str, chunk_table: str, embedding_table: str, vector_size: int, recreate: bool) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        if recreate:
            log.info("Dropping existing vote tables before recreate")
            cur.execute(f"DROP TABLE IF EXISTS {schema}.{embedding_table} CASCADE")
            cur.execute(f"DROP TABLE IF EXISTS {schema}.{chunk_table} CASCADE")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{chunk_table} (
              id BIGSERIAL PRIMARY KEY,
              source_hash TEXT NOT NULL UNIQUE,
              vote_id TEXT NOT NULL,
              congress INTEGER NOT NULL,
              chamber TEXT NOT NULL,
              session TEXT NOT NULL,
              number INTEGER NOT NULL,
              vote_date TIMESTAMPTZ,
              category TEXT,
              vote_type TEXT,
              question TEXT,
              subject TEXT,
              result TEXT,
              bill_id TEXT,
              body TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              chunk_index INTEGER NOT NULL DEFAULT 0,
              content_hash TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{embedding_table} (
              chunk_id BIGINT PRIMARY KEY REFERENCES {schema}.{chunk_table}(id) ON DELETE CASCADE,
              embedding vector({vector_size}) NOT NULL,
              model TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        for sql in [
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_vote_id_idx ON {schema}.{chunk_table} (vote_id)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_bill_id_idx ON {schema}.{chunk_table} (bill_id)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_congress_idx ON {schema}.{chunk_table} (congress)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_chamber_idx ON {schema}.{chunk_table} (chamber)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_date_idx ON {schema}.{chunk_table} (vote_date)",
            f"CREATE INDEX IF NOT EXISTS {chunk_table}_source_hash_idx ON {schema}.{chunk_table} (source_hash)",
        ]:
            cur.execute(sql)


def ensure_embedding_dimensions(conn, schema: str, embedding_table: str, vector_size: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s
              AND c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (schema, embedding_table),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Could not find {schema}.{embedding_table}.embedding")
        actual = row[0] or ""
        expected = f"vector({vector_size})"
        if actual != expected:
            raise RuntimeError(
                f"{schema}.{embedding_table}.embedding is {actual}, expected {expected}. "
                "Use --recreate or point at a matching table."
            )


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------

def stage_shard(cur, rows: list[tuple], batch_size: int) -> None:
    cur.execute(
        """
        CREATE TEMP TABLE staging_bill_chunks (
          source_hash TEXT,
          bill_id TEXT,
          canonical_bill_id TEXT,
          congress INTEGER,
          bill_type TEXT,
          bill_number TEXT,
          chunk_type TEXT,
          section_path TEXT,
          title TEXT,
          body TEXT,
          token_count INTEGER,
          source_version TEXT,
          status TEXT,
          section_enum TEXT,
          section_header TEXT,
          chunk_index INTEGER,
          original_chunk_index INTEGER,
          document_text_hash TEXT,
          section_text_hash TEXT,
          document_text_hashes JSONB,
          document_aliases JSONB,
          section_aliases JSONB,
          embedding_text TEXT,
          model TEXT
        ) ON COMMIT DROP
        """
    )
    execute_values(
        cur,
        """
        INSERT INTO staging_bill_chunks (
          source_hash, bill_id, canonical_bill_id, congress, bill_type, bill_number,
          chunk_type, section_path, title, body, token_count, source_version, status,
          section_enum, section_header, chunk_index, original_chunk_index,
          document_text_hash, section_text_hash, document_text_hashes,
          document_aliases, section_aliases, embedding_text, model
        ) VALUES %s
        """,
        rows,
        page_size=batch_size,
    )


def stage_vote_shard(cur, rows: list[tuple], batch_size: int) -> None:
    cur.execute(
        """
        CREATE TEMP TABLE staging_vote_chunks (
          source_hash TEXT,
          vote_id TEXT,
          congress INTEGER,
          chamber TEXT,
          session TEXT,
          number INTEGER,
          vote_date TIMESTAMPTZ,
          category TEXT,
          vote_type TEXT,
          question TEXT,
          subject TEXT,
          result TEXT,
          bill_id TEXT,
          body TEXT,
          token_count INTEGER,
          chunk_index INTEGER,
          content_hash TEXT,
          embedding_text TEXT,
          model TEXT
        ) ON COMMIT DROP
        """
    )
    execute_values(
        cur,
        """
        INSERT INTO staging_vote_chunks (
          source_hash, vote_id, congress, chamber, session, number, vote_date,
          category, vote_type, question, subject, result, bill_id, body,
          token_count, chunk_index, content_hash, embedding_text, model
        ) VALUES %s
        """,
        rows,
        page_size=batch_size,
    )


def replace_shard(conn, schema: str, chunk_table: str, embedding_table: str, shard_rows: list[tuple], batch_size: int) -> None:
    if not shard_rows:
        return

    bill_ids = sorted({row[1] for row in shard_rows if row[1]})
    with conn:
        with conn.cursor() as cur:
            stage_shard(cur, shard_rows, batch_size=batch_size)
            cur.execute(
                f"DELETE FROM {schema}.{chunk_table} WHERE bill_id = ANY(%s)",
                (bill_ids,),
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.{chunk_table} (
                  source_hash, bill_id, canonical_bill_id, congress, bill_type, bill_number,
                  chunk_type, section_path, title, body, token_count, source_version, status,
                  section_enum, section_header, chunk_index, original_chunk_index,
                  document_text_hash, section_text_hash, document_text_hashes,
                  document_aliases, section_aliases
                )
                SELECT
                  source_hash, bill_id, canonical_bill_id, congress, bill_type, bill_number,
                  chunk_type, section_path, title, body, token_count, source_version, status,
                  section_enum, section_header, chunk_index, original_chunk_index,
                  document_text_hash, section_text_hash, document_text_hashes,
                  document_aliases, section_aliases
                FROM staging_bill_chunks
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.{embedding_table} (chunk_id, embedding, model)
                SELECT c.id, s.embedding_text::vector, s.model
                FROM {schema}.{chunk_table} c
                JOIN staging_bill_chunks s ON s.source_hash = c.source_hash
                WHERE c.bill_id = ANY(%s)
                ON CONFLICT (chunk_id) DO UPDATE
                  SET embedding = EXCLUDED.embedding,
                      model = EXCLUDED.model
                """,
                (bill_ids,),
            )


def replace_vote_shard(conn, schema: str, chunk_table: str, embedding_table: str, shard_rows: list[tuple], batch_size: int) -> None:
    if not shard_rows:
        return

    source_hashes = sorted({row[0] for row in shard_rows if row[0]})
    with conn:
        with conn.cursor() as cur:
            stage_vote_shard(cur, shard_rows, batch_size=batch_size)
            cur.execute(
                f"DELETE FROM {schema}.{chunk_table} WHERE source_hash = ANY(%s)",
                (source_hashes,),
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.{chunk_table} (
                  source_hash, vote_id, congress, chamber, session, number, vote_date,
                  category, vote_type, question, subject, result, bill_id, body,
                  token_count, chunk_index, content_hash
                )
                SELECT
                  source_hash, vote_id, congress, chamber, session, number, vote_date,
                  category, vote_type, question, subject, result, bill_id, body,
                  token_count, chunk_index, content_hash
                FROM staging_vote_chunks
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.{embedding_table} (chunk_id, embedding, model)
                SELECT c.id, s.embedding_text::vector, s.model
                FROM {schema}.{chunk_table} c
                JOIN staging_vote_chunks s ON s.source_hash = c.source_hash
                WHERE c.source_hash = ANY(%s)
                ON CONFLICT (chunk_id) DO UPDATE
                  SET embedding = EXCLUDED.embedding,
                      model = EXCLUDED.model
                """,
                (source_hashes,),
            )


def verify_shard_rows(shard_path: Path, chunks: list[dict]) -> tuple[list[dict], int]:
    rows = []
    tokens = 0
    for chunk in chunks:
        if "embedding" not in chunk:
            raise RuntimeError(f"Missing embedding in shard {shard_path}")
        rows.append(chunk)
        tokens += safe_int(chunk.get("tokens"))
    return rows, tokens


def verify_vote_shard_rows(shard_path: Path, chunks: list[dict]) -> tuple[list[dict], int]:
    rows = []
    tokens = 0
    for chunk in chunks:
        if "embedding" not in chunk:
            raise RuntimeError(f"Missing embedding in shard {shard_path}")
        if "source_hash" not in chunk:
            raise RuntimeError(f"Missing source_hash in shard {shard_path}")
        rows.append(chunk)
        tokens += safe_int(chunk.get("token_count", chunk.get("tokens")), 0)
    return rows, tokens


# ---------------------------------------------------------------------------
# Post-load: HNSW index + verification
# ---------------------------------------------------------------------------

def build_hnsw_index(
    conn,
    schema: str,
    embedding_table: str,
    m: int,
    ef_construction: int,
    maintenance_work_mem: str,
    max_parallel_maintenance_workers: int,
) -> None:
    """Build the HNSW index in a single bulk pass after all data is loaded."""
    index_name = f"{embedding_table}_embedding_hnsw_idx"
    log.info(
        "Building HNSW index "
        f"(m={m}, ef_construction={ef_construction}, "
        f"maintenance_work_mem={maintenance_work_mem}, "
        f"max_parallel_maintenance_workers={max_parallel_maintenance_workers}) "
        "— this will take a while..."
    )

    # Drop any partial index from a previous attempt
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS nlp.{index_name}")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = %s", (maintenance_work_mem,))
        cur.execute("SET max_parallel_maintenance_workers = %s", (max_parallel_maintenance_workers,))
        cur.execute(
            f"""
            CREATE INDEX {index_name}
              ON {schema}.{embedding_table}
              USING hnsw (embedding vector_cosine_ops)
              WITH (m = {m}, ef_construction = {ef_construction})
            """
        )
    conn.commit()
    log.info("HNSW index built successfully")


def verify_counts(conn, schema: str, chunk_table: str, embedding_table: str) -> None:
    """Log row counts for both tables and warn if they diverge."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.{chunk_table}")
        chunks = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {schema}.{embedding_table}")
        embeddings = cur.fetchone()[0]

    log.info(f"{'=' * 60}")
    log.info(f"Verification: {chunks:,} chunks | {embeddings:,} embeddings")
    if chunks != embeddings:
        log.warning(f"Count mismatch — {chunks - embeddings:,} chunks are missing embeddings")
    else:
        log.info("Counts match")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load embedded chunk shards into PostgreSQL pgvector tables")
    parser.add_argument("--mode", choices=["bills", "votes"], default="bills",
                        help="Load bill or vote embedding tables (default: bills)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input embedded shard directory (default depends on --mode)")
    parser.add_argument("--dsn", type=str, default=DEFAULT_DSN,
                        help="PostgreSQL connection string (default: PG_CONNECTION_STRING env var)")
    parser.add_argument("--create-db", action="store_true",
                        help="Create the target database and enable pgvector if they don't exist")
    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA,
                        help=f"Target schema (default: {DEFAULT_SCHEMA})")
    parser.add_argument("--chunk-table", type=str, default=None,
                        help="Chunk table name (default depends on --mode)")
    parser.add_argument("--embedding-table", type=str, default=None,
                        help="Embedding table name (default depends on --mode)")
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE,
                        help=f"Vector size (default: {DEFAULT_VECTOR_SIZE})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Embedding model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Rows per bulk insert into staging (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M,
                        help=f"HNSW m parameter — connections per node (default: {DEFAULT_HNSW_M})")
    parser.add_argument("--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION,
                        help=f"HNSW ef_construction parameter (default: {DEFAULT_HNSW_EF_CONSTRUCTION})")
    parser.add_argument("--maintenance-work-mem", type=str, default=DEFAULT_MAINTENANCE_WORK_MEM,
                        help=f"maintenance_work_mem to use during HNSW build (default: {DEFAULT_MAINTENANCE_WORK_MEM})")
    parser.add_argument("--max-parallel-maintenance-workers", type=int, default=DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS,
                        help=f"max_parallel_maintenance_workers during HNSW build (default: {DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS})")
    parser.add_argument("--skip-hnsw", action="store_true",
                        help="Skip HNSW index build after loading (useful for partial loads)")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip shard scanning/loading and only build the HNSW index + verify counts")
    parser.add_argument("--include-aliases", action="store_true",
                        help="Include document_aliases and section_aliases in stored chunk rows")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate tables before loading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and validate shards without touching PostgreSQL")
    args = parser.parse_args()

    if not args.dsn:
        log.error("PostgreSQL DSN not provided. Set PG_CONNECTION_STRING or pass --dsn.")
        return

    if args.input is None:
        args.input = str(INPUT_DIR if args.mode == "bills" else DATA_DIR / "embedded_vote_chunks")
    if args.chunk_table is None:
        args.chunk_table = DEFAULT_CHUNK_TABLE if args.mode == "bills" else DEFAULT_VOTE_CHUNK_TABLE
    if args.embedding_table is None:
        args.embedding_table = DEFAULT_EMBEDDING_TABLE if args.mode == "bills" else DEFAULT_VOTE_EMBEDDING_TABLE

    input_path = Path(args.input)
    shard_paths: list[Path] = []
    if not args.index_only:
        if not input_path.exists():
            log.error(f"Input path not found: {input_path}")
            return

        try:
            shard_paths = iter_embedded_shards(input_path)
        except ValueError as e:
            log.error(str(e))
            return

        if not shard_paths:
            log.error(f"No embedded shard files found under {input_path}")
            return

    mode = "index-only" if args.index_only else "full-load"
    log.info(
        f"Starting pgvector upserter | mode={mode} | load={args.mode} | input={input_path} | schema={args.schema} | "
        f"tables={args.schema}.{args.chunk_table},{args.schema}.{args.embedding_table}"
    )

    total_chunks = 0
    total_tokens = 0

    if not args.index_only:
        log.info(f"Found {len(shard_paths)} embedded shard file(s); scanning...")

    # Create database if requested
    if args.create_db:
        create_database_if_not_exists(args.dsn)

    conn = psycopg2.connect(args.dsn)
    try:
        # Schema setup
        if args.mode == "votes":
            ensure_vote_schema(conn, args.schema, args.chunk_table, args.embedding_table, args.vector_size, args.recreate)
        else:
            ensure_schema(conn, args.schema, args.chunk_table, args.embedding_table, args.vector_size, args.recreate)
        ensure_embedding_dimensions(conn, args.schema, args.embedding_table, args.vector_size)
        conn.commit()

        if args.index_only:
            if args.skip_hnsw:
                log.info("Index-only mode requested with --skip-hnsw; skipping index build")
            else:
                build_hnsw_index(
                    conn,
                    args.schema,
                    args.embedding_table,
                    args.hnsw_m,
                    args.hnsw_ef_construction,
                    args.maintenance_work_mem,
                    args.max_parallel_maintenance_workers,
                )
            verify_counts(conn, args.schema, args.chunk_table, args.embedding_table)
            return

        # Scan and load one shard at a time to keep memory bounded.
        upserted = 0
        for shard_index, shard_path in enumerate(shard_paths, 1):
            log.info(f"Scanning shard {shard_index}/{len(shard_paths)}: {shard_path.name}")
            chunk_dicts = read_jsonl(shard_path)
            if args.mode == "votes":
                rows, shard_tokens = verify_vote_shard_rows(shard_path, chunk_dicts)
                shard_rows = [
                    build_vote_row(chunk=chunk, model=args.model)
                    for chunk in rows
                ]
            else:
                rows, shard_tokens = verify_shard_rows(shard_path, chunk_dicts)
                shard_rows = [
                    build_chunk_row(chunk=chunk, model=args.model, include_aliases=args.include_aliases)
                    for chunk in rows
                ]
            total_chunks += len(shard_rows)
            total_tokens += shard_tokens
            log.info(f"  {len(shard_rows)} rows, {shard_tokens:,} tokens")

            if args.dry_run:
                continue

            log.info(f"Loading shard {shard_index}/{len(shard_paths)}: {shard_path.name} ({len(shard_rows)} rows)")
            if args.mode == "votes":
                replace_vote_shard(
                    conn=conn,
                    schema=args.schema,
                    chunk_table=args.chunk_table,
                    embedding_table=args.embedding_table,
                    shard_rows=shard_rows,
                    batch_size=args.batch_size,
                )
            else:
                replace_shard(
                    conn=conn,
                    schema=args.schema,
                    chunk_table=args.chunk_table,
                    embedding_table=args.embedding_table,
                    shard_rows=shard_rows,
                    batch_size=args.batch_size,
                )
            upserted += len(shard_rows)
            log.info(f"  Shard {shard_index}/{len(shard_paths)} done: {upserted:,} total rows loaded")

        log.info(f"Total: {len(shard_paths)} shards | {total_chunks:,} chunks | {total_tokens:,} tokens")

        if args.dry_run:
            log.info("[DRY RUN] Exiting without touching PostgreSQL")
            return

        log.info(f"{'=' * 60}")
        log.info(f"Load complete: {upserted:,} chunks into {args.schema}.{args.chunk_table}")

        # Build HNSW index
        if not args.skip_hnsw:
            build_hnsw_index(
                conn,
                args.schema,
                args.embedding_table,
                args.hnsw_m,
                args.hnsw_ef_construction,
                args.maintenance_work_mem,
                args.max_parallel_maintenance_workers,
            )

        # Verify
        verify_counts(conn, args.schema, args.chunk_table, args.embedding_table)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
