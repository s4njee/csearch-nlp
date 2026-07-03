#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tiktoken",
# ]
# ///
from __future__ import annotations
"""
chunker.py — Parse downloaded bill text into semantically meaningful chunks.

Handles two formats:
  - XML bills (majority): parses <section> tags, extracts structure
  - HTML/text bills (fallback): splits on "SECTION" headers in plain text

The chunker now performs two exact deduplication passes before embedding:
  1. Full-document dedup across identical normalized bill texts
  2. Section-level dedup across identical normalized section bodies

Each emitted chunk keeps alias metadata so duplicate bills/statuses can still
be shown later even though only one canonical chunk is embedded.

Usage:
    python chunker.py                              # process all congresses
    python chunker.py --congresses 110             # specific congress
    python chunker.py --limit 10                   # first 10 bills only
    python chunker.py --max-tokens 256             # smaller chunks
"""

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_BASE = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent / "data"))
OUTPUT_DIR = DATA_BASE
SHARD_OUTPUT_DIR = OUTPUT_DIR / "processed_chunks"

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
ENCODING_NAME = "cl100k_base"  # used by text-embedding-3-small
DEFAULT_SHARD_BILLS = 500
DEFAULT_SHARD_CHUNKS = 40000

# Boilerplate section headers to skip
BOILERPLATE = {
    "short title",
    "short title; table of contents",
    "table of contents",
    "effective date",
    "effective dates",
    "severability",
    "severability clause",
    "definitions",  # often needed for context but too generic alone
}

# Bill type display names
TYPE_DISPLAY = {
    "hr": "H.R.",
    "s": "S.",
    "hjres": "H.J.Res.",
    "sjres": "S.J.Res.",
    "hconres": "H.Con.Res.",
    "sconres": "S.Con.Res.",
    "hres": "H.Res.",
    "sres": "S.Res.",
}

VERSION_PRIORITY = {
    "enr": 0,
    "eas": 1,
    "eah": 1,
    "es": 2,
    "eh": 2,
    "rs": 3,
    "rh": 3,
    "rds": 4,
    "rfh": 4,
    "rfs": 4,
    "ih": 5,
    "is": 5,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chunker")

# Try to load tiktoken encoding; fall back to char-based estimation
enc = None
if _tiktoken_available:
    try:
        enc = tiktoken.get_encoding(ENCODING_NAME)
        log.info("Using tiktoken for tokenization")
    except Exception:
        pass

if enc is None:
    log.warning("tiktoken unavailable — using ~4 chars/token estimate")


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(len(text.split()) / 1.3))


def _encode(text: str) -> list:
    if enc is not None:
        return enc.encode(text)
    return text.split()


def _decode(tokens: list) -> str:
    if enc is not None:
        return enc.decode(tokens)
    return " ".join(tokens)


def split_with_overlap(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into chunks of ~max_tokens with overlap, at sentence boundaries."""
    tokens = _encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = _decode(tokens[start:end])

        if end < len(tokens):
            last_period = chunk_text.rfind(". ")
            if last_period > len(chunk_text) // 2:
                chunk_text = chunk_text[:last_period + 1]
                end = start + len(_encode(chunk_text))

        chunks.append(chunk_text.strip())
        start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Normalization / hashing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and clean up extracted text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    """Normalize text for exact-match hashing without dropping legal punctuation."""
    return clean_text(text).lower()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def version_rank(version: str) -> tuple[int, str]:
    v = (version or "").lower()
    return (VERSION_PRIORITY.get(v, 99), v)


def alias_sort_key(alias: dict) -> tuple:
    return (
        version_rank(alias.get("version", "")),
        alias.get("bill_id", ""),
        str(alias.get("section_enum", "")),
        alias.get("section_header", ""),
    )


def canonical_bill_sort_key(chunk_or_alias: dict) -> tuple:
    """Sort canonical bills by congress, type, then numeric bill number when possible."""
    number = str(chunk_or_alias.get("number", ""))
    try:
        number_key = (0, int(number))
    except ValueError:
        number_key = (1, number)
    return (
        int(chunk_or_alias.get("congress", 0)),
        chunk_or_alias.get("type", ""),
        number_key,
        chunk_or_alias.get("canonical_bill_id", chunk_or_alias.get("bill_id", "")),
    )


def dedupe_records(records: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    seen = set()
    unique = []
    for record in sorted(records, key=alias_sort_key):
        key = tuple(record.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def extract_text(elem) -> str:
    """Recursively extract all text from an XML element, collapsing whitespace."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def extract_bill_metadata_xml(root) -> dict:
    """Extract metadata from the XML bill root element."""
    meta = {}
    meta["bill_stage"] = root.get("bill-stage", "")

    form = root.find(".//form")
    if form is not None:
        legis_num = form.find("legis-num")
        if legis_num is not None and legis_num.text:
            meta["legis_num"] = clean_text(legis_num.text)

        official = form.find("official-title")
        if official is not None:
            meta["official_title"] = clean_text(extract_text(official))

        congress_el = form.find("congress")
        if congress_el is not None and congress_el.text:
            meta["congress_text"] = clean_text(congress_el.text)

    dc_title = root.find(".//{http://purl.org/dc/elements/1.1/}title")
    if dc_title is not None and dc_title.text:
        meta["dc_title"] = clean_text(dc_title.text)

    dc_date = root.find(".//{http://purl.org/dc/elements/1.1/}date")
    if dc_date is not None and dc_date.text:
        meta["date"] = dc_date.text.strip()

    return meta


def build_section_body_xml(sec) -> str:
    """Extract section body text while excluding the section enum/header tags."""
    parts = []
    for child in sec:
        if child.tag in {"enum", "header"}:
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return clean_text(" ".join(parts))


def parse_sections_xml(root) -> list[dict]:
    """Extract sections from XML bill."""
    sections = []

    for sec in root.iter("section"):
        enum_el = sec.find("enum")
        header_el = sec.find("header")

        enum = clean_text(enum_el.text) if enum_el is not None and enum_el.text else ""
        header = clean_text(extract_text(header_el)) if header_el is not None else ""

        if header.lower().strip() in BOILERPLATE:
            continue

        body_text = build_section_body_xml(sec)
        if not body_text:
            continue

        subsections = list(sec.iter("subsection"))
        if subsections and len(subsections) > 1:
            sub_texts = []
            for sub in subsections:
                sub_text = clean_text(extract_text(sub))
                if sub_text:
                    sub_texts.append(sub_text)
            sections.append({
                "enum": enum,
                "header": header,
                "body_text": body_text,
                "sub_texts": sub_texts,
            })
        else:
            sections.append({
                "enum": enum,
                "header": header,
                "body_text": body_text,
                "sub_texts": [],
            })

    return sections


# ---------------------------------------------------------------------------
# Text/HTML helpers
# ---------------------------------------------------------------------------

def parse_sections_text(content: str) -> list[dict]:
    """Parse plain-text/HTML-derived bill content into section records."""
    section_pattern = re.compile(
        r"(?:^|\n)\s*(SECTION|SEC\.?)\s+(\d+[A-Za-z]?)\.\s*(.*?)(?=\n)",
        re.IGNORECASE,
    )
    parts = section_pattern.split(content)

    if len(parts) <= 1:
        text = clean_text(content)
        if not text or len(text) < 50:
            return []
        return [{
            "enum": "",
            "header": "",
            "body_text": text,
            "sub_texts": [],
        }]

    sections = []
    i = 1
    while i + 2 < len(parts):
        sec_num = parts[i + 1]
        sec_header = parts[i + 2].strip()
        sec_body = parts[i + 3] if i + 3 < len(parts) else ""
        i += 4

        header_clean = clean_text(sec_header)
        if header_clean.lower().rstrip(".") in BOILERPLATE:
            continue

        body_text = clean_text(sec_body)
        if not body_text:
            continue

        sections.append({
            "enum": sec_num,
            "header": header_clean,
            "body_text": body_text,
            "sub_texts": [],
        })

    return sections


# ---------------------------------------------------------------------------
# Document loading / dedup
# ---------------------------------------------------------------------------

def build_document_alias(meta: dict, short_title: str) -> dict:
    congress = meta.get("congress", "?")
    btype = meta.get("type", "hr")
    number = meta.get("number", "?")
    return {
        "bill_id": meta.get("bill_id", f"{btype}{number}-{congress}"),
        "congress": congress,
        "type": btype,
        "number": number,
        "short_title": short_title,
        "version": meta.get("version", ""),
        "status": meta.get("status", ""),
        "format": meta.get("format", ""),
    }


def load_bill_document(filepath: Path, meta: dict) -> dict | None:
    """Load one bill into a parsed document record."""
    fmt = meta.get("format", "")

    if fmt == "xml":
        try:
            tree = ET.parse(filepath)
        except ET.ParseError as e:
            log.warning(f"XML parse error in {filepath.name}: {e}")
            return None

        root = tree.getroot()
        bill_meta = extract_bill_metadata_xml(root)
        sections = parse_sections_xml(root)
        full_text = clean_text(extract_text(root))
        short_title = meta.get("short_title", "") or bill_meta.get("official_title", "")
    else:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        if fmt == "html":
            content = re.sub(r"<[^>]+>", "", content)
        full_text = clean_text(content)
        sections = parse_sections_text(content)
        short_title = meta.get("short_title", "") or meta.get("official_title", "")

    if not full_text:
        return None

    alias = build_document_alias(meta, short_title)
    return {
        "bill_id": alias["bill_id"],
        "congress": alias["congress"],
        "type": alias["type"],
        "number": alias["number"],
        "short_title": short_title,
        "version": alias["version"],
        "status": alias["status"],
        "format": alias["format"],
        "full_text": full_text,
        "document_text_hash": hash_text(normalize_for_hash(full_text)),
        "sections": sections,
        "alias": alias,
    }


def build_canonical_documents(documents: list[dict]) -> tuple[list[dict], int]:
    """Collapse exact full-document duplicates into canonical documents."""
    grouped = defaultdict(list)
    for doc in documents:
        grouped[doc["document_text_hash"]].append(doc)

    canonical_documents = []
    duplicate_docs = 0
    for document_hash, docs in grouped.items():
        docs.sort(key=lambda d: alias_sort_key(d["alias"]))
        canonical_doc = docs[0]
        document_aliases = dedupe_records([doc["alias"] for doc in docs], ("bill_id",))
        duplicate_docs += max(0, len(docs) - 1)
        canonical_documents.append({
            **canonical_doc,
            "canonical_bill_id": canonical_doc["bill_id"],
            "document_aliases": document_aliases,
            "document_text_hash": document_hash,
        })

    canonical_documents.sort(key=lambda d: alias_sort_key(d["alias"]))
    return canonical_documents, duplicate_docs


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def build_chunk_prefix(alias: dict, section_enum: str, section_header: str) -> str:
    display_type = TYPE_DISPLAY.get(alias.get("type", "hr"), alias.get("type", "hr").upper())
    number = alias.get("number", "?")
    congress = alias.get("congress", "?")
    if section_enum:
        return f"[{display_type} {number}, {congress}th Congress] Section {section_enum}: {section_header} —"
    if section_header:
        return f"[{display_type} {number}, {congress}th Congress] {section_header} —"
    return f"[{display_type} {number}, {congress}th Congress] —"


def build_section_chunks(section: dict, canonical_alias: dict, max_tokens: int, overlap: int) -> list[dict]:
    """Chunk a canonical section body while preserving bill/section context."""
    enum = section.get("section_enum", "").rstrip(".")
    header = section.get("section_header", "")
    body_text = section.get("section_body", "")
    prefix = build_chunk_prefix(canonical_alias, enum, header)

    if not body_text:
        return []

    chunks = []
    token_count = count_tokens(f"{prefix} {body_text}")

    if token_count <= max_tokens:
        chunks.append({
            "chunk_index": 0,
            "text": f"{prefix} {body_text}",
            "tokens": token_count,
        })
        return chunks

    if section.get("sub_texts"):
        current_chunk = prefix
        chunk_idx = 0
        for sub_text in section["sub_texts"]:
            candidate = f"{current_chunk} {sub_text}" if current_chunk != prefix else f"{prefix} {sub_text}"
            if count_tokens(candidate) > max_tokens and current_chunk != prefix:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": current_chunk.strip(),
                    "tokens": count_tokens(current_chunk),
                })
                chunk_idx += 1
                current_chunk = f"{prefix} {sub_text}"
            else:
                current_chunk = candidate

            if count_tokens(current_chunk) > max_tokens:
                for sc in split_with_overlap(current_chunk, max_tokens, overlap):
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "text": sc.strip(),
                        "tokens": count_tokens(sc),
                    })
                    chunk_idx += 1
                current_chunk = prefix

        if current_chunk != prefix:
            chunks.append({
                "chunk_index": chunk_idx,
                "text": current_chunk.strip(),
                "tokens": count_tokens(current_chunk),
            })
        return chunks

    for i, sc in enumerate(split_with_overlap(f"{prefix} {body_text}", max_tokens, overlap)):
        chunks.append({
            "chunk_index": i,
            "text": sc.strip(),
            "tokens": count_tokens(sc),
        })
    return chunks


def build_canonical_chunks(canonical_documents: list[dict], max_tokens: int, overlap: int) -> tuple[list[dict], int]:
    """Collapse exact duplicate sections across canonical docs and emit chunks."""
    section_groups = defaultdict(list)

    for doc in canonical_documents:
        for section in doc["sections"]:
            section_body = section.get("body_text", "")
            if not section_body:
                continue

            section_hash = hash_text(normalize_for_hash(section_body))
            section_aliases = []
            for alias in doc["document_aliases"]:
                section_aliases.append({
                    **alias,
                    "section_enum": section.get("enum", "").rstrip("."),
                    "section_header": section.get("header", ""),
                })

            section_groups[section_hash].append({
                "canonical_bill_id": doc["canonical_bill_id"],
                "document_text_hash": doc["document_text_hash"],
                "document_aliases": doc["document_aliases"],
                "section_aliases": section_aliases,
                "section_enum": section.get("enum", "").rstrip("."),
                "section_header": section.get("header", ""),
                "section_body": section_body,
                "sub_texts": section.get("sub_texts", []),
                "short_title": doc.get("short_title", ""),
            })

    chunks = []
    duplicate_sections = 0

    for section_hash, records in section_groups.items():
        all_section_aliases = dedupe_records(
            [alias for record in records for alias in record["section_aliases"]],
            ("bill_id", "section_enum", "section_header"),
        )
        all_document_aliases = dedupe_records(
            [alias for record in records for alias in record["document_aliases"]],
            ("bill_id",),
        )
        duplicate_sections += max(0, len(records) - 1)

        canonical_section_alias = min(all_section_aliases, key=alias_sort_key)
        canonical_section = min(
            records,
            key=lambda record: (
                0 if any(
                    alias["bill_id"] == canonical_section_alias["bill_id"]
                    and alias.get("section_enum", "") == record.get("section_enum", "")
                    and alias.get("section_header", "") == record.get("section_header", "")
                    for alias in record["section_aliases"]
                ) else 1,
                record.get("section_enum", ""),
                record.get("section_header", ""),
                record.get("document_text_hash", ""),
            ),
        )

        for chunk in build_section_chunks(canonical_section, canonical_section_alias, max_tokens, overlap):
            chunks.append({
                "bill_id": canonical_section_alias["bill_id"],
                "canonical_bill_id": canonical_section_alias["bill_id"],
                "congress": canonical_section_alias["congress"],
                "type": canonical_section_alias["type"],
                "number": canonical_section_alias["number"],
                "short_title": canonical_section_alias.get("short_title", canonical_section.get("short_title", "")),
                "version": canonical_section_alias.get("version", ""),
                "status": canonical_section_alias.get("status", ""),
                "section_enum": canonical_section.get("section_enum", ""),
                "section_header": canonical_section.get("section_header", ""),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "tokens": chunk["tokens"],
                "document_text_hash": canonical_section["document_text_hash"],
                "document_text_hashes": sorted({record["document_text_hash"] for record in records}),
                "section_text_hash": section_hash,
                "document_aliases": all_document_aliases,
                "section_aliases": all_section_aliases,
            })

    chunks.sort(key=lambda c: (alias_sort_key(c), c.get("chunk_index", 0)))
    return chunks, duplicate_sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_bill_dirs() -> list[Path]:
    return sorted(d for d in DATA_BASE.iterdir() if d.is_dir() and d.name.startswith("bills_"))


def iter_meta_files(bills_dir: Path) -> list[Path]:
    """Return bill metadata files, excluding hidden/macOS artifact files."""
    return sorted(
        path for path in bills_dir.rglob("*.meta.json")
        if not any(part.startswith(".") for part in path.parts)
        and not path.name.startswith("._")
    )


def load_bill_ids(path: Path) -> set[str]:
    """Load changed bill IDs from a JSON list or content_hasher change manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if isinstance(payload, dict):
        return {str(item) for item in payload.get("changed_bill_ids", [])}
    raise ValueError(f"Unsupported bill-id manifest format: {path}")


def bill_id_meta_path(bills_dir: Path, bill_id: str) -> Path | None:
    match = re.fullmatch(r"([a-z]+)(\d+)-(\d+)", bill_id.lower())
    if not match:
        return None
    btype, number, congress = match.groups()
    if congress != bills_dir.name.split("_", 1)[1]:
        return None
    return bills_dir / btype / f"{btype}{number}.meta.json"


def meta_files_for_bill_ids(bills_dir: Path, bill_ids: set[str]) -> list[Path]:
    """Resolve bill IDs to metadata files without scanning the whole congress."""
    paths = []
    missing = []
    for bill_id in sorted(bill_ids):
        path = bill_id_meta_path(bills_dir, bill_id)
        if path is not None and path.exists():
            paths.append(path)
        else:
            missing.append(bill_id)
    if missing:
        # A changed bill we can't resolve to a meta file must NOT be silently
        # dropped: the run would still succeed and promote the hash manifest,
        # marking that bill "done" though it was never embedded. Degrade to a
        # full-congress scan so every changed bill is processed; the (rare)
        # cost is one non-incremental run, loudly flagged.
        log.error(
            "Could not resolve %s of %s changed bill(s) to a metadata file "
            "(e.g. %s) — falling back to a full-congress scan so no changed "
            "bill is skipped.",
            len(missing), len(bill_ids), ", ".join(missing[:5]),
        )
        return iter_meta_files(bills_dir)
    return paths


def apply_chunk_filters(chunks: list[dict], min_tokens: int, cap: int) -> tuple[list[dict], int, int]:
    """Apply global minimum-token and per-canonical-bill chunk cap filters."""
    before = len(chunks)
    filtered = [c for c in chunks if c["tokens"] >= min_tokens]
    below_min = before - len(filtered)

    if cap <= 0:
        return renumber_chunk_indexes(filtered), below_min, 0

    by_bill = defaultdict(list)
    for chunk in filtered:
        by_bill[chunk["canonical_bill_id"]].append(chunk)

    capped_chunks = []
    total_dropped = 0
    for canonical_bill_id, bill_chunks in by_bill.items():
        if len(bill_chunks) <= cap:
            capped_chunks.extend(bill_chunks)
            continue

        by_section = defaultdict(list)
        for chunk in bill_chunks:
            section_key = (
                chunk.get("section_text_hash", ""),
                chunk.get("section_enum", ""),
                chunk.get("section_header", ""),
            )
            by_section[section_key].append(chunk)

        must_keep = []
        remaining = []
        for section_chunks in by_section.values():
            section_chunks.sort(
                key=lambda c: (-c["tokens"], c.get("chunk_index", 0))
            )
            must_keep.append(section_chunks[0])
            remaining.extend(section_chunks[1:])

        if len(must_keep) >= cap:
            must_keep.sort(
                key=lambda c: (-c["tokens"], c.get("section_enum", ""), c.get("chunk_index", 0))
            )
            kept = must_keep[:cap]
        else:
            remaining.sort(
                key=lambda c: (-c["tokens"], c.get("section_enum", ""), c.get("chunk_index", 0))
            )
            kept = must_keep + remaining[:cap - len(must_keep)]

        kept.sort(key=lambda c: (c.get("section_enum", ""), c.get("chunk_index", 0)))
        capped_chunks.extend(kept)
        dropped = len(bill_chunks) - cap
        total_dropped += dropped
        log.info(f"  Capped {canonical_bill_id}: {len(bill_chunks)} → {cap} chunks ({dropped} dropped)")

    return renumber_chunk_indexes(capped_chunks), below_min, total_dropped


def renumber_chunk_indexes(chunks: list[dict]) -> list[dict]:
    """Reassign contiguous chunk indexes after filtering/capping, preserving the original value."""
    grouped = defaultdict(list)
    for chunk in chunks:
        section_key = (
            chunk["canonical_bill_id"],
            chunk.get("section_text_hash", ""),
            chunk.get("section_enum", ""),
            chunk.get("section_header", ""),
        )
        grouped[section_key].append(chunk)

    renumbered = []
    for section_key, section_chunks in grouped.items():
        section_chunks.sort(key=lambda c: (c.get("section_enum", ""), c.get("chunk_index", 0)))
        for new_index, chunk in enumerate(section_chunks):
            updated = dict(chunk)
            updated["original_chunk_index"] = chunk.get("original_chunk_index", chunk.get("chunk_index", 0))
            updated["chunk_index"] = new_index
            renumbered.append(updated)

    renumbered.sort(key=lambda c: (canonical_bill_sort_key(c), c.get("section_enum", ""), c.get("chunk_index", 0)))
    return renumbered


def write_congress_shards(
    congress_num: str,
    chunks: list[dict],
    stats: dict,
    shard_bill_limit: int,
    shard_chunk_limit: int,
    output_root: Path,
) -> dict:
    """Write one congress of canonical chunks to rolling JSONL shards and a manifest."""
    congress_dir = output_root / str(congress_num)
    congress_dir.mkdir(parents=True, exist_ok=True)

    for path in congress_dir.glob("shard-*.jsonl"):
        path.unlink()
    manifest_path = congress_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    by_bill = defaultdict(list)
    for chunk in chunks:
        by_bill[chunk["canonical_bill_id"]].append(chunk)

    ordered_bill_ids = sorted(
        by_bill.keys(),
        key=lambda bill_id: canonical_bill_sort_key(by_bill[bill_id][0]),
    )
    manifest = []
    shard_index = 1
    current_bill_ids = []
    current_chunks = []

    def flush_current() -> None:
        nonlocal shard_index, current_bill_ids, current_chunks
        if not current_chunks:
            return

        shard_path = congress_dir / f"shard-{shard_index:05d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            for chunk in current_chunks:
                f.write(json.dumps(chunk, separators=(",", ":")))
                f.write("\n")

        manifest.append({
            "shard_path": str(shard_path),
            "canonical_bill_count": len(current_bill_ids),
            "chunk_count": len(current_chunks),
            "first_canonical_bill_id": current_bill_ids[0],
            "last_canonical_bill_id": current_bill_ids[-1],
        })
        shard_index += 1
        current_bill_ids = []
        current_chunks = []

    for bill_id in ordered_bill_ids:
        bill_chunks = sorted(by_bill[bill_id], key=lambda c: (c.get("section_enum", ""), c.get("chunk_index", 0)))
        would_exceed_bills = shard_bill_limit > 0 and len(current_bill_ids) >= shard_bill_limit
        would_exceed_chunks = shard_chunk_limit > 0 and current_chunks and (len(current_chunks) + len(bill_chunks) > shard_chunk_limit)
        if would_exceed_bills or would_exceed_chunks:
            flush_current()

        current_bill_ids.append(bill_id)
        current_chunks.extend(bill_chunks)

    flush_current()

    manifest_output = {
        "congress": int(congress_num),
        "shard_bill_limit": shard_bill_limit,
        "shard_chunk_limit": shard_chunk_limit,
        "raw_documents": stats.get("raw_documents", 0),
        "canonical_documents": stats.get("canonical_documents", 0),
        "duplicate_documents": stats.get("duplicate_documents", 0),
        "duplicate_sections": stats.get("duplicate_sections", 0),
        "filtered_below_min_tokens": stats.get("filtered_below_min_tokens", 0),
        "dropped_by_bill_cap": stats.get("dropped_by_bill_cap", 0),
        "chunks_written": len(chunks),
        "shard_count": len(manifest),
        "shards": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_output, indent=2), encoding="utf-8")
    return manifest_output


def process_congress(
    bills_dir: Path,
    max_tokens: int,
    overlap: int,
    limit: int = 0,
    bill_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Process all downloaded bills in a congress directory."""
    meta_files = meta_files_for_bill_ids(bills_dir, bill_ids) if bill_ids is not None else iter_meta_files(bills_dir)
    if limit > 0:
        meta_files = meta_files[:limit]

    documents = []
    for mf in meta_files:
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning(f"Skipping unreadable metadata file {mf}: {exc}")
            continue
        fmt = meta.get("format", "")
        stem = mf.stem.replace(".meta", "")
        parent = mf.parent

        if fmt == "xml":
            content_file = parent / f"{stem}.xml"
        elif fmt == "html":
            content_file = parent / f"{stem}.htm"
        else:
            content_file = parent / f"{stem}.txt"

        if not content_file.exists():
            continue

        document = load_bill_document(content_file, meta)
        if document is not None:
            documents.append(document)

    canonical_documents, duplicate_docs = build_canonical_documents(documents)
    chunks, duplicate_sections = build_canonical_chunks(canonical_documents, max_tokens, overlap)
    return chunks, {
        "raw_documents": len(documents),
        "canonical_documents": len(canonical_documents),
        "duplicate_documents": duplicate_docs,
        "duplicate_sections": duplicate_sections,
    }


def main():
    parser = argparse.ArgumentParser(description="Chunk downloaded bill text for embedding")
    parser.add_argument("--congresses", nargs="+", type=int, default=None,
                        help="Congress numbers to process (default: all available)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens per chunk (default: {MAX_TOKENS})")
    parser.add_argument("--overlap", type=int, default=OVERLAP_TOKENS,
                        help=f"Token overlap between split chunks (default: {OVERLAP_TOKENS})")
    parser.add_argument("--max-chunks-per-bill", type=int, default=200,
                        help="Cap chunks per canonical bill, keeping the most substantive (default: 200, 0 = no cap)")
    parser.add_argument("--shard-bills", type=int, default=DEFAULT_SHARD_BILLS,
                        help=f"Canonical bills per JSONL shard (default: {DEFAULT_SHARD_BILLS})")
    parser.add_argument("--shard-chunks", type=int, default=DEFAULT_SHARD_CHUNKS,
                        help=f"Soft chunk cap per JSONL shard, applied between bills (default: {DEFAULT_SHARD_CHUNKS})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max bills to process per congress (0 = all)")
    parser.add_argument("--bill-ids-file", type=str, default=None,
                        help="JSON list or content_hasher change manifest of bill IDs to chunk")
    parser.add_argument("--output-dir", type=str, default=str(SHARD_OUTPUT_DIR),
                        help=f"Processed shard output root (default: {SHARD_OUTPUT_DIR})")
    args = parser.parse_args()

    bill_ids = load_bill_ids(Path(args.bill_ids_file)) if args.bill_ids_file else None
    output_root = Path(args.output_dir)
    if bill_ids is not None:
        log.info("Incremental chunking enabled for %s bill(s)", len(bill_ids))

    bill_dirs = find_bill_dirs()
    if args.congresses:
        bill_dirs = [d for d in bill_dirs if int(d.name.split("_")[1]) in args.congresses]

    if not bill_dirs:
        log.error("No downloaded bill directories found. Run fetcher.py first.")
        return

    total_stats = defaultdict(int)
    total_chunks = 0
    total_shards = 0
    total_tokens = 0
    canonical_bill_ids = set()
    for bills_dir in bill_dirs:
        congress_num = bills_dir.name.split("_")[1]
        log.info(f"Processing congress {congress_num}...")
        chunks, stats = process_congress(bills_dir, args.max_tokens, args.overlap, args.limit, bill_ids=bill_ids)
        chunks, below_min, capped = apply_chunk_filters(chunks, min_tokens=30, cap=args.max_chunks_per_bill)
        log.info(
            "  → %s chunks | docs %s -> %s | dropped %s duplicate docs | dropped %s duplicate sections",
            len(chunks),
            stats["raw_documents"],
            stats["canonical_documents"],
            stats["duplicate_documents"],
            stats["duplicate_sections"],
        )
        if below_min:
            log.info(f"  → filtered out {below_min} chunks below 30 tokens")
        if capped:
            log.info(f"  → dropped {capped} chunks via per-bill cap")
        stats["filtered_below_min_tokens"] = below_min
        stats["dropped_by_bill_cap"] = capped
        for key, value in stats.items():
            total_stats[key] += value
        stats["chunks_written"] = len(chunks)
        manifest = write_congress_shards(
            congress_num=congress_num,
            chunks=chunks,
            stats=stats,
            shard_bill_limit=args.shard_bills,
            shard_chunk_limit=args.shard_chunks,
            output_root=output_root,
        )
        total_chunks += len(chunks)
        total_shards += manifest["shard_count"]
        total_tokens += sum(chunk["tokens"] for chunk in chunks)
        canonical_bill_ids.update(chunk["canonical_bill_id"] for chunk in chunks)
        log.info(f"  → wrote {manifest['shard_count']} shard(s) to {output_root / congress_num}")

    if total_chunks:
        token_values = []
        for bills_dir in bill_dirs:
            congress_num = bills_dir.name.split("_")[1]
            manifest_path = output_root / congress_num / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            for shard in manifest.get("shards", []):
                shard_path = Path(shard["shard_path"])
                with shard_path.open(encoding="utf-8") as f:
                    for line in f:
                        token_values.append(json.loads(line)["tokens"])
        log.info(f"{'=' * 60}")
        log.info(
            "Total: %s chunks across %s shard(s) from %s canonical bills (%s raw docs, %s canonical docs)",
            total_chunks,
            total_shards,
            len(canonical_bill_ids),
            total_stats["raw_documents"],
            total_stats["canonical_documents"],
        )
        log.info(
            "Dedup removed %s duplicate documents and %s duplicate sections",
            total_stats["duplicate_documents"],
            total_stats["duplicate_sections"],
        )
        log.info(f"Tokens: min={min(token_values)}, avg={sum(token_values)//len(token_values)}, max={max(token_values)}")
        est_cost = total_tokens / 1_000_000 * 0.02
        log.info(f"Estimated embedding cost: ${est_cost:.4f}")
    log.info(f"Saved JSONL shards under {output_root}")


if __name__ == "__main__":
    main()
