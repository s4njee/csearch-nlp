#!/usr/bin/env python3
"""
content_hasher.py — Detect meaningful bill text changes between pipeline runs.

Scans fetched bill XML/HTML/TXT files for a congress, extracts legislative text
(stripping XML attributes and metadata that change without affecting content —
e.g. effective dates, action dates, print numbers), and compares SHA256 hashes
against a stored manifest from the previous run.

Bills where only XML metadata changed are treated as unchanged and excluded from
the manifest diff. Only bills with actual text changes incur re-chunking and
re-embedding costs.

Exit codes:
    0 — one or more bills have changed content (proceed with pipeline)
    1 — no content changes detected (skip pipeline)

Usage:
    python content_hasher.py --congress 119
    python content_hasher.py --congress 119 --data-dir /app/data --verbose
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_DIR = DATA_DIR / "hash_manifests"
BILLS_DIR = DATA_DIR / "bills"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("content_hasher")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_xml_text(content: bytes) -> str:
    """
    Extract legislative text from XML, ignoring all attributes.

    Attributes (effective-date, action-date, print-number, etc.) are stripped
    so that metadata-only updates don't produce a different hash.
    """
    try:
        root = ET.fromstring(content)
        parts = []
        for elem in root.iter():
            if elem.text:
                t = elem.text.strip()
                if t:
                    parts.append(t)
            if elem.tail:
                t = elem.tail.strip()
                if t:
                    parts.append(t)
        return " ".join(parts)
    except ET.ParseError:
        # Malformed XML — fall back to stripping tags with regex
        return extract_html_text(content)


def extract_html_text(content: bytes) -> str:
    """Strip HTML/XML tags and return plain text."""
    text = content.decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(path: Path) -> str:
    """Dispatch to the appropriate extractor based on file extension."""
    content = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return extract_xml_text(content)
    elif suffix in (".htm", ".html"):
        return extract_html_text(content)
    else:
        return content.decode("utf-8", errors="replace")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning(f"Could not read manifest at {manifest_path} — treating all bills as new")
        return {}


def save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect meaningful bill text changes for a congress"
    )
    parser.add_argument("--congress", type=int, required=True,
                        help="Congress number to check (e.g. 119)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help=f"Root data directory (default: {DATA_DIR})")
    parser.add_argument("--verbose", action="store_true",
                        help="Log each changed/unchanged bill")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bills_dir = data_dir / f"bills_{args.congress}"
    manifest_path = data_dir / "hash_manifests" / f"{args.congress}.json"

    if not bills_dir.exists():
        log.error(f"Bills directory not found: {bills_dir}")
        return 1

    old_manifest = load_manifest(manifest_path)
    new_manifest: dict[str, str] = {}

    changed = []
    unchanged = []

    for meta_path in sorted(bills_dir.rglob("*.meta.json")):
        stem = meta_path.stem.replace(".meta", "")
        bill_dir = meta_path.parent

        # Find the content file (prefer XML, then HTML, then TXT)
        content_path = None
        for ext in (".xml", ".htm", ".txt"):
            candidate = bill_dir / f"{stem}{ext}"
            if candidate.exists():
                content_path = candidate
                break

        if content_path is None:
            continue

        # Load bill_id from meta
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bill_id = meta.get("bill_id", stem)
        except (json.JSONDecodeError, OSError):
            bill_id = stem

        try:
            text = extract_text(content_path)
            current_hash = hash_text(text)
        except OSError as exc:
            log.warning(f"Could not read {content_path}: {exc}")
            continue

        new_manifest[bill_id] = current_hash

        if old_manifest.get(bill_id) != current_hash:
            changed.append(bill_id)
            if args.verbose:
                was = "new" if bill_id not in old_manifest else "modified"
                log.info(f"  {bill_id}: {was}")
        else:
            unchanged.append(bill_id)
            if args.verbose:
                log.info(f"  {bill_id}: unchanged")

    log.info(
        f"Congress {args.congress}: {len(changed)} changed, "
        f"{len(unchanged)} unchanged, {len(new_manifest)} total"
    )

    save_manifest(manifest_path, new_manifest)
    log.info(f"Hash manifest updated: {manifest_path}")

    if changed:
        log.info(f"Changes detected — pipeline should proceed")
        return 0
    else:
        log.info("No content changes — pipeline can be skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
