#!/usr/bin/env python3
"""
chunker.py — Parse downloaded bill text into semantically meaningful chunks.

Handles two formats:
  - XML bills (majority): parses <section> tags, extracts structure
  - HTML/text bills (fallback): splits on "SECTION" headers in plain text

Each chunk gets a context prefix:
  "[H.R. 1424, 110th Congress] Section 101: Purchases of Troubled Assets —"

Sections exceeding ~512 tokens are split at <subsection>/<paragraph>
boundaries with 64-token overlap.

Boilerplate sections (Short Title, Effective Date, Severability, etc.)
are discarded.

Usage:
    python chunker.py                              # process all congresses
    python chunker.py --congresses 110             # specific congress
    python chunker.py --limit 10                   # first 10 bills only
    python chunker.py --max-tokens 256             # smaller chunks
"""

import argparse
import json
import re
import logging
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

DATA_BASE = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
ENCODING_NAME = "cl100k_base"  # used by text-embedding-3-small

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
    # ~1.3 words per token for English legal text
    return max(1, int(len(text.split()) / 1.3))


def _encode(text: str) -> list:
    if enc is not None:
        return enc.encode(text)
    # Fallback: split on whitespace (avg ~1.3 words per token for English)
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

        # Try to break at sentence boundary (last period before limit)
        if end < len(tokens):
            last_period = chunk_text.rfind(". ")
            if last_period > len(chunk_text) // 2:
                chunk_text = chunk_text[:last_period + 1]
                end = start + len(_encode(chunk_text))

        chunks.append(chunk_text.strip())
        start = max(end - overlap, start + 1)  # ensure forward progress

    return chunks


# ---------------------------------------------------------------------------
# XML bill parsing
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


def clean_text(text: str) -> str:
    """Normalize whitespace and clean up extracted text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    return text.strip()


def extract_bill_metadata_xml(root) -> dict:
    """Extract metadata from the XML bill root element."""
    meta = {}

    # Bill type and stage from attributes
    meta["bill_stage"] = root.get("bill-stage", "")

    # Form section has legis-num, official-title, congress, session
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

    # Dublin Core metadata
    dc_title = root.find(".//{http://purl.org/dc/elements/1.1/}title")
    if dc_title is not None and dc_title.text:
        meta["dc_title"] = clean_text(dc_title.text)

    dc_date = root.find(".//{http://purl.org/dc/elements/1.1/}date")
    if dc_date is not None and dc_date.text:
        meta["date"] = dc_date.text.strip()

    return meta


def parse_sections_xml(root) -> list[dict]:
    """Extract sections from XML bill, returning list of {enum, header, text, subsections}."""
    sections = []

    for sec in root.iter("section"):
        enum_el = sec.find("enum")
        header_el = sec.find("header")

        enum = clean_text(enum_el.text) if enum_el is not None and enum_el.text else ""
        header = clean_text(extract_text(header_el)) if header_el is not None else ""

        # Check for boilerplate
        if header.lower().strip() in BOILERPLATE:
            continue

        # Try to split at subsection boundaries first
        subsections = list(sec.iter("subsection"))
        if subsections and len(subsections) > 1:
            # Section has subsections — chunk at that boundary
            sub_texts = []
            for sub in subsections:
                sub_text = clean_text(extract_text(sub))
                if sub_text:
                    sub_texts.append(sub_text)
            sections.append({
                "enum": enum,
                "header": header,
                "text": clean_text(extract_text(sec)),
                "sub_texts": sub_texts,
            })
        else:
            # Flat section — just grab all text
            full_text = clean_text(extract_text(sec))
            sections.append({
                "enum": enum,
                "header": header,
                "text": full_text,
                "sub_texts": [],
            })

    return sections


def chunk_xml_bill(filepath: Path, meta: dict, max_tokens: int, overlap: int) -> list[dict]:
    """Parse an XML bill and return a list of chunks with metadata."""
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        log.warning(f"XML parse error in {filepath.name}: {e}")
        return []

    root = tree.getroot()
    bill_meta = extract_bill_metadata_xml(root)
    sections = parse_sections_xml(root)

    congress = meta.get("congress", "?")
    btype = meta.get("type", "hr")
    number = meta.get("number", "?")
    bill_id = meta.get("bill_id", f"{btype}{number}-{congress}")
    display_type = TYPE_DISPLAY.get(btype, btype.upper())
    short_title = meta.get("short_title", "") or bill_meta.get("official_title", "")

    chunks = []
    for sec in sections:
        enum = sec["enum"].rstrip(".")
        header = sec["header"]
        prefix = f"[{display_type} {number}, {congress}th Congress] Section {enum}: {header} —"

        full_text = sec["text"]
        token_count = count_tokens(f"{prefix} {full_text}")

        if token_count <= max_tokens:
            # Fits in one chunk
            chunks.append({
                "bill_id": bill_id,
                "congress": congress,
                "type": btype,
                "number": number,
                "short_title": short_title,
                "section_enum": enum,
                "section_header": header,
                "chunk_index": 0,
                "text": f"{prefix} {full_text}",
                "tokens": token_count,
            })
        elif sec["sub_texts"]:
            # Split at subsection boundaries
            current_chunk = prefix
            chunk_idx = 0
            for sub_text in sec["sub_texts"]:
                candidate = f"{current_chunk} {sub_text}" if current_chunk != prefix else f"{prefix} {sub_text}"
                if count_tokens(candidate) > max_tokens and current_chunk != prefix:
                    # Flush current chunk
                    chunks.append({
                        "bill_id": bill_id,
                        "congress": congress,
                        "type": btype,
                        "number": number,
                        "short_title": short_title,
                        "section_enum": enum,
                        "section_header": header,
                        "chunk_index": chunk_idx,
                        "text": current_chunk.strip(),
                        "tokens": count_tokens(current_chunk),
                    })
                    chunk_idx += 1
                    current_chunk = f"{prefix} {sub_text}"
                else:
                    current_chunk = candidate

                # If single subsection exceeds limit, force-split it
                if count_tokens(current_chunk) > max_tokens:
                    sub_chunks = split_with_overlap(current_chunk, max_tokens, overlap)
                    for sc in sub_chunks:
                        chunks.append({
                            "bill_id": bill_id,
                            "congress": congress,
                            "type": btype,
                            "number": number,
                            "short_title": short_title,
                            "section_enum": enum,
                            "section_header": header,
                            "chunk_index": chunk_idx,
                            "text": sc.strip(),
                            "tokens": count_tokens(sc),
                        })
                        chunk_idx += 1
                    current_chunk = prefix

            # Flush remaining
            if current_chunk != prefix:
                chunks.append({
                    "bill_id": bill_id,
                    "congress": congress,
                    "type": btype,
                    "number": number,
                    "short_title": short_title,
                    "section_enum": enum,
                    "section_header": header,
                    "chunk_index": chunk_idx,
                    "text": current_chunk.strip(),
                    "tokens": count_tokens(current_chunk),
                })
        else:
            # No subsections — force-split with overlap
            sub_chunks = split_with_overlap(f"{prefix} {full_text}", max_tokens, overlap)
            for i, sc in enumerate(sub_chunks):
                chunks.append({
                    "bill_id": bill_id,
                    "congress": congress,
                    "type": btype,
                    "number": number,
                    "short_title": short_title,
                    "section_enum": enum,
                    "section_header": header,
                    "chunk_index": i,
                    "text": sc.strip(),
                    "tokens": count_tokens(sc),
                })

    return chunks


# ---------------------------------------------------------------------------
# HTML/text bill parsing (fallback)
# ---------------------------------------------------------------------------

def chunk_text_bill(filepath: Path, meta: dict, max_tokens: int, overlap: int) -> list[dict]:
    """Parse a plain-text/HTML bill and return chunks split on SECTION headers."""
    content = filepath.read_text(encoding="utf-8", errors="replace")

    # Strip HTML wrapper if present
    content = re.sub(r"<[^>]+>", "", content)

    congress = meta.get("congress", "?")
    btype = meta.get("type", "hr")
    number = meta.get("number", "?")
    bill_id = meta.get("bill_id", f"{btype}{number}-{congress}")
    display_type = TYPE_DISPLAY.get(btype, btype.upper())
    short_title = meta.get("short_title", "")

    # Split on section headers like "SECTION 1." or "SEC. 101."
    section_pattern = re.compile(
        r"(?:^|\n)\s*(SECTION|SEC\.?)\s+(\d+[A-Za-z]?)\.\s*(.*?)(?=\n)",
        re.IGNORECASE,
    )

    parts = section_pattern.split(content)

    # If no sections found, treat whole text as one chunk
    if len(parts) <= 1:
        text = clean_text(content)
        if not text or len(text) < 50:
            return []
        prefix = f"[{display_type} {number}, {congress}th Congress] —"
        sub_chunks = split_with_overlap(f"{prefix} {text}", max_tokens, overlap)
        return [{
            "bill_id": bill_id,
            "congress": congress,
            "type": btype,
            "number": number,
            "short_title": short_title,
            "section_enum": "",
            "section_header": "",
            "chunk_index": i,
            "text": sc.strip(),
            "tokens": count_tokens(sc),
        } for i, sc in enumerate(sub_chunks)]

    chunks = []
    # parts structure: [preamble, "SECTION", num, header, body, "SEC", num, header, body, ...]
    i = 1  # skip preamble
    while i + 2 < len(parts):
        sec_type = parts[i]      # "SECTION" or "SEC"
        sec_num = parts[i + 1]   # "1", "101", etc.
        sec_header = parts[i + 2].strip()
        sec_body = parts[i + 3] if i + 3 < len(parts) else ""
        i += 4

        header_clean = clean_text(sec_header)
        if header_clean.lower().rstrip(".") in BOILERPLATE:
            continue

        text = clean_text(f"{sec_header} {sec_body}")
        prefix = f"[{display_type} {number}, {congress}th Congress] Section {sec_num}: {header_clean} —"

        sub_chunks = split_with_overlap(f"{prefix} {text}", max_tokens, overlap)
        for ci, sc in enumerate(sub_chunks):
            chunks.append({
                "bill_id": bill_id,
                "congress": congress,
                "type": btype,
                "number": number,
                "short_title": short_title,
                "section_enum": sec_num,
                "section_header": header_clean,
                "chunk_index": ci,
                "text": sc.strip(),
                "tokens": count_tokens(sc),
            })

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_bill_dirs() -> list[Path]:
    """Find all bills_* directories under DATA_BASE."""
    return sorted(d for d in DATA_BASE.iterdir() if d.is_dir() and d.name.startswith("bills_"))


def process_congress(bills_dir: Path, max_tokens: int, overlap: int, limit: int = 0) -> list[dict]:
    """Process all downloaded bills in a congress directory."""
    meta_files = sorted(bills_dir.rglob("*.meta.json"))
    if limit > 0:
        meta_files = meta_files[:limit]

    all_chunks = []
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        fmt = meta.get("format", "")
        stem = mf.stem.replace(".meta", "")
        parent = mf.parent

        # Find the content file
        if fmt == "xml":
            content_file = parent / f"{stem}.xml"
        elif fmt == "html":
            content_file = parent / f"{stem}.htm"
        else:
            content_file = parent / f"{stem}.txt"

        if not content_file.exists():
            continue

        if fmt == "xml":
            chunks = chunk_xml_bill(content_file, meta, max_tokens, overlap)
        else:
            chunks = chunk_text_bill(content_file, meta, max_tokens, overlap)

        all_chunks.extend(chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk downloaded bill text for embedding")
    parser.add_argument("--congresses", nargs="+", type=int, default=None,
                        help="Congress numbers to process (default: all available)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens per chunk (default: {MAX_TOKENS})")
    parser.add_argument("--overlap", type=int, default=OVERLAP_TOKENS,
                        help=f"Token overlap between split chunks (default: {OVERLAP_TOKENS})")
    parser.add_argument("--max-chunks-per-bill", type=int, default=200,
                        help="Cap chunks per bill, keeping the most substantive (default: 200, 0 = no cap)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max bills to process per congress (0 = all)")
    args = parser.parse_args()

    bill_dirs = find_bill_dirs()
    if args.congresses:
        bill_dirs = [d for d in bill_dirs if int(d.name.split("_")[1]) in args.congresses]

    if not bill_dirs:
        log.error("No downloaded bill directories found. Run fetcher.py first.")
        return

    all_chunks = []
    for bills_dir in bill_dirs:
        congress_num = bills_dir.name.split("_")[1]
        log.info(f"Processing congress {congress_num}...")
        chunks = process_congress(bills_dir, args.max_tokens, args.overlap, args.limit)
        log.info(f"  → {len(chunks)} chunks from {bills_dir.name}")
        all_chunks.extend(chunks)

    # Filter out tiny/empty chunks (< 30 tokens)
    MIN_TOKENS = 30
    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if c["tokens"] >= MIN_TOKENS]
    if before != len(all_chunks):
        log.info(f"Filtered out {before - len(all_chunks)} chunks below {MIN_TOKENS} tokens")

    # Per-bill chunk cap: keep the most substantive chunks
    cap = args.max_chunks_per_bill
    if cap > 0:
        from collections import defaultdict
        by_bill = defaultdict(list)
        for c in all_chunks:
            by_bill[c["bill_id"]].append(c)

        capped_chunks = []
        total_dropped = 0
        for bill_id, bill_chunks in by_bill.items():
            if len(bill_chunks) <= cap:
                capped_chunks.extend(bill_chunks)
            else:
                # Sort by token count descending — keep the meatiest chunks
                bill_chunks.sort(key=lambda c: c["tokens"], reverse=True)
                kept = bill_chunks[:cap]
                # Re-sort by section order for consistent output
                kept.sort(key=lambda c: (c["section_enum"], c["chunk_index"]))
                capped_chunks.extend(kept)
                dropped = len(bill_chunks) - cap
                total_dropped += dropped
                log.info(f"  Capped {bill_id}: {len(bill_chunks)} → {cap} chunks ({dropped} dropped)")

        if total_dropped:
            log.info(f"Per-bill cap ({cap}): dropped {total_dropped} chunks total")
        all_chunks = capped_chunks

    # Stats
    if all_chunks:
        tokens = [c["tokens"] for c in all_chunks]
        bills = len(set(c["bill_id"] for c in all_chunks))
        log.info(f"{'='*60}")
        log.info(f"Total: {len(all_chunks)} chunks from {bills} bills")
        log.info(f"Tokens: min={min(tokens)}, avg={sum(tokens)//len(tokens)}, max={max(tokens)}")
        est_cost = sum(tokens) / 1_000_000 * 0.02  # text-embedding-3-small pricing
        log.info(f"Estimated embedding cost: ${est_cost:.4f}")

    # Write output
    out_path = OUTPUT_DIR / "processed_chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
