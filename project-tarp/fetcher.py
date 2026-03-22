#!/usr/bin/env python3
"""
fetcher.py — Download full bill text from GovInfo for any congress.

Auto-discovers congress directories under the project root (e.g. 110/, 111/,
112/) from the @unitedstates/congress scraper, then fetches the best available
text (XML > HTML > plain text) for each bill from GovInfo's content endpoint.

GovInfo returns HTTP 200 with an HTML error page for missing content (no
real 404), so we validate response bodies before saving.

Usage:
    python fetcher.py                              # all congresses found locally
    python fetcher.py --congresses 110 111         # specific congresses
    python fetcher.py --bill-types hr s            # only House and Senate bills
    python fetcher.py --limit 10                   # first 10 bills per congress
    python fetcher.py --dry-run                    # print plan, don't download
    python fetcher.py --clean                      # remove saved error pages
"""

import argparse
import json
import time
import logging
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # csearch-nlp/
OUTPUT_BASE = Path(__file__).resolve().parent / "data"  # project-tarp/data/

ALL_BILL_TYPES = ["hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"]

GOVINFO_BASE = "https://www.govinfo.gov/content/pkg"

# Version lists by bill status — avoids trying 20+ versions per bill.
VERSIONS_ENACTED = ["enr", "eas", "eah", "es", "eh", "ih", "is"]
VERSIONS_PASSED = ["es", "eh", "eas", "eah", "cps", "cph", "ih", "is"]
VERSIONS_REPORTED = ["rs", "rh", "rds", "rfh", "rfs", "ih", "is"]
VERSIONS_REFERRED = ["rfh", "rfs", "ih", "is"]
VERSIONS_INTRODUCED = ["ih", "is"]
VERSIONS_FALLBACK = ["enr", "eh", "es", "rh", "rs", "ih", "is"]

HEADERS = {
    "User-Agent": "project-tarp/0.1 (congressional research; contact: sanjee.yogeswaran@gmail.com)",
}

RATE_LIMIT_SECONDS = 0.35  # ~3 req/s

# Strings that indicate a GovInfo soft-404 (HTTP 200 but error page body)
ERROR_SIGNATURES = [
    "Page Not Found",
    "page you requested cannot be found",
    "govinfo.gov/error",
]

# GovInfo URL subdirectory → file extension mapping
FORMAT_MAP = {
    "xml":  ("xml",  "xml"),
    "html": ("html", "htm"),
    "text": ("text", "txt"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetcher")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_versions(bill_status: str, bill_type: str) -> list[str]:
    """Pick a short version list based on how far the bill progressed."""
    s = bill_status.upper()
    if "ENACTED" in s or "SIGNED" in s:
        versions = VERSIONS_ENACTED
    elif "PASS" in s:
        versions = VERSIONS_PASSED
    elif "REPORTED" in s:
        versions = VERSIONS_REPORTED
    elif "REFERRED" in s:
        versions = VERSIONS_REFERRED
    elif "INTRODUCED" in s:
        versions = VERSIONS_INTRODUCED
    else:
        versions = VERSIONS_FALLBACK

    chamber = "h" if bill_type in ("hr", "hres", "hjres", "hconres") else "s"
    filtered = [v for v in versions if v == "enr" or v.endswith(chamber)]
    return filtered if filtered else versions


def build_url(congress: int, btype: str, number: str, version: str, fmt: str = "xml") -> str:
    """Construct a GovInfo content URL for a specific bill version + format."""
    pkg = f"BILLS-{congress}{btype}{number}{version}"
    subdir, ext = FORMAT_MAP[fmt]
    return f"{GOVINFO_BASE}/{pkg}/{subdir}/{pkg}.{ext}"


def is_error_page(content: str) -> bool:
    """Detect GovInfo soft-404 pages (HTTP 200 but error body)."""
    head = content[:2048]
    return any(sig in head for sig in ERROR_SIGNATURES)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_congresses(root: Path) -> list[int]:
    """Auto-detect congress directories (numeric names) under root."""
    congresses = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.isdigit():
            bills_dir = d / "bills"
            if bills_dir.is_dir():
                congresses.append(int(d.name))
    return congresses


def discover_bills(congress: int, bill_types: list[str], data_root: Path = PROJECT_ROOT) -> list[dict]:
    """Walk {congress}/bills/{type}/{type}{number}/ and return a manifest."""
    bills_root = data_root / str(congress) / "bills"
    bills = []
    for btype in bill_types:
        type_dir = bills_root / btype
        if not type_dir.is_dir():
            continue
        for bill_dir in sorted(type_dir.iterdir()):
            if not bill_dir.is_dir():
                continue
            name = bill_dir.name
            number = name.replace(btype, "", 1)
            if not number.isdigit():
                continue

            meta = {}
            data_json = bill_dir / "data.json"
            if data_json.exists():
                try:
                    with open(data_json) as f:
                        raw = json.load(f)
                    meta = {
                        "bill_id": raw.get("bill_id", f"{btype}{number}-{congress}"),
                        "short_title": raw.get("short_title", ""),
                        "official_title": raw.get("official_title", ""),
                        "status": raw.get("status", ""),
                    }
                except (json.JSONDecodeError, KeyError):
                    meta = {"bill_id": f"{btype}{number}-{congress}"}

            bills.append({
                "type": btype,
                "number": number,
                "congress": congress,
                "local_dir": str(bill_dir),
                **meta,
            })
    return bills


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def fetch_bill_text(bill: dict, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Try to download the best available text version for a bill.
    Returns a result dict with status and metadata.
    """
    congress = bill["congress"]
    btype = bill["type"]
    number = bill["number"]
    bill_id = bill.get("bill_id", f"{btype}{number}-{congress}")

    # Check if already downloaded (any format)
    meta_file = output_dir / btype / f"{btype}{number}.meta.json"
    if meta_file.exists():
        return {"bill_id": bill_id, "status": "skipped", "reason": "already exists"}

    versions = pick_versions(bill.get("status", ""), btype)

    if dry_run:
        url = build_url(congress, btype, number, versions[0])
        return {"bill_id": bill_id, "status": "dry_run", "url_sample": url, "versions": versions}

    # Try each version: xml first (has section structure), then html, text
    for version in versions:
        for fmt in ["xml", "html", "text"]:
            url = build_url(congress, btype, number, version, fmt)
            try:
                req = Request(url, headers=HEADERS)
                with urlopen(req, timeout=30) as resp:
                    content = resp.read().decode("utf-8", errors="replace")

                if is_error_page(content):
                    time.sleep(RATE_LIMIT_SECONDS)
                    continue

                # Valid content — write it
                _, ext = FORMAT_MAP[fmt]
                out_file = output_dir / btype / f"{btype}{number}.{ext}"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(content, encoding="utf-8")

                meta = {
                    "bill_id": bill_id,
                    "congress": congress,
                    "type": btype,
                    "number": number,
                    "version": version,
                    "format": fmt,
                    "source_url": url,
                    "short_title": bill.get("short_title", ""),
                    "official_title": bill.get("official_title", ""),
                    "status": bill.get("status", ""),
                    "bytes": len(content),
                }
                meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                time.sleep(RATE_LIMIT_SECONDS)
                return {"bill_id": bill_id, "status": "ok", "version": version, "fmt": fmt, "bytes": len(content)}

            except HTTPError as e:
                if e.code == 404:
                    continue
                elif e.code == 429:
                    log.warning(f"Rate limited on {bill_id}, sleeping 10s...")
                    time.sleep(10)
                    continue
                else:
                    return {"bill_id": bill_id, "status": "error", "error": f"HTTP {e.code}: {url}"}
            except (URLError, TimeoutError) as e:
                return {"bill_id": bill_id, "status": "error", "error": str(e)}

    return {"bill_id": bill_id, "status": "not_found", "reason": "no version matched"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch congressional bill text from GovInfo")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory containing congress folders (default: parent of project-tarp)")
    parser.add_argument("--congresses", nargs="+", type=int, default=None,
                        help="Congress numbers to fetch (default: auto-detect all)")
    parser.add_argument("--min-congress", type=int, default=103,
                        help="Skip congresses before this (default: 103, earliest with text on GovInfo)")
    parser.add_argument("--bill-types", nargs="+", default=["hr", "s"],
                        help="Bill types to fetch (default: hr s)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max bills to process per congress (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without downloading")
    parser.add_argument("--clean", action="store_true",
                        help="Remove previously downloaded error pages, then exit")
    args = parser.parse_args()

    # Determine data root
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT
    data_root = data_root.resolve()

    # Determine which congresses to process
    if args.congresses:
        congresses = sorted(args.congresses)
    else:
        congresses = find_congresses(data_root)
        if not congresses:
            log.error(f"No congress directories found under {data_root}")
            return

    # Filter by min-congress (GovInfo text starts at 103rd, 1993)
    skipped = [c for c in congresses if c < args.min_congress]
    congresses = [c for c in congresses if c >= args.min_congress]
    if skipped:
        log.info(f"Skipping congresses {skipped[0]}-{skipped[-1]} (before {args.min_congress}, no text on GovInfo)")
    log.info(f"Auto-detected {len(congresses)} congresses in {data_root}: {congresses[0]}-{congresses[-1]}")

    log.info(f"Congresses: {congresses} | Bill types: {args.bill_types}")

    # Clean mode
    if args.clean:
        total_removed = 0
        for congress in congresses:
            output_dir = OUTPUT_BASE / f"bills_{congress}"
            if not output_dir.exists():
                continue
            removed = 0
            for meta_path in output_dir.rglob("*.meta.json"):
                text_stem = meta_path.stem.replace(".meta", "")
                parent = meta_path.parent
                content_file = None
                for ext in ["xml", "htm", "txt"]:
                    candidate = parent / f"{text_stem}.{ext}"
                    if candidate.exists():
                        content_file = candidate
                        break
                if content_file and is_error_page(content_file.read_text(encoding="utf-8", errors="replace")):
                    content_file.unlink()
                    meta_path.unlink()
                    removed += 1
            if removed:
                log.info(f"Congress {congress}: cleaned {removed} error pages")
            total_removed += removed
        log.info(f"Total cleaned: {total_removed}")
        return

    # Process each congress
    grand_stats = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}

    for congress in congresses:
        output_dir = OUTPUT_BASE / f"bills_{congress}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"{'='*60}")
        log.info(f"Congress {congress}")
        log.info(f"{'='*60}")

        bills = discover_bills(congress, args.bill_types, data_root)
        log.info(f"Found {len(bills)} bills")

        if args.limit > 0:
            bills = bills[:args.limit]
            log.info(f"Limited to {args.limit} bills")

        if args.dry_run:
            for b in bills[:5]:
                result = fetch_bill_text(b, output_dir, dry_run=True)
                log.info(f"[DRY RUN] {result}")
            if len(bills) > 5:
                log.info(f"... and {len(bills) - 5} more")
            continue

        stats = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}
        results_log = []

        for i, bill in enumerate(bills, 1):
            result = fetch_bill_text(bill, output_dir)
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            results_log.append(result)

            if status == "ok":
                log.info(f"[{congress}] [{i}/{len(bills)}] ✓ {result['bill_id']} ({result['version']}.{result['fmt']}, {result['bytes']:,}B)")
            elif status == "skipped":
                log.debug(f"[{congress}] [{i}/{len(bills)}] ⏭ {result['bill_id']}")
            elif status == "not_found":
                log.warning(f"[{congress}] [{i}/{len(bills)}] ✗ {result['bill_id']} — no text on GovInfo")
            else:
                log.error(f"[{congress}] [{i}/{len(bills)}] ✗ {result['bill_id']} — {result.get('error', 'unknown')}")

            if i % 100 == 0:
                log.info(f"[{congress}] Progress: {i}/{len(bills)} | ok={stats['ok']} skip={stats['skipped']} miss={stats['not_found']} err={stats['error']}")

        # Per-congress report
        log.info(f"[{congress}] Done: ok={stats['ok']} skip={stats['skipped']} miss={stats['not_found']} err={stats['error']}")

        # Save per-congress manifest
        manifest_path = output_dir / "fetch_manifest.json"
        manifest_path.write_text(json.dumps(results_log, indent=2), encoding="utf-8")

        for k in grand_stats:
            grand_stats[k] += stats.get(k, 0)

    # Grand total
    if len(congresses) > 1 and not args.dry_run:
        log.info(f"{'='*60}")
        log.info(f"ALL CONGRESSES: ok={grand_stats['ok']} skip={grand_stats['skipped']} miss={grand_stats['not_found']} err={grand_stats['error']}")


if __name__ == "__main__":
    main()
