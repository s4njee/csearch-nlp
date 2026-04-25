#!/usr/bin/env bash
# nightly_update.sh — Incremental bill fetch, chunk, embed, and upsert pipeline.
#
# Environment variables:
#   CONGRESS              Congress number to process (default: 119)
#   CONGRESS_DATA_ROOT    Path to @unitedstates/congress scraper output (default: /root/congress/data)
#   DATA_DIR              Path to TARP working data (default: /app/data)
#   PG_CONNECTION_STRING  PostgreSQL DSN (required)
#   OPENAI_API_KEY        OpenAI API key (required for embedding backend)
#
set -euo pipefail

CONGRESS="${CONGRESS:-119}"
CONGRESS_DATA_ROOT="${CONGRESS_DATA_ROOT:-/root/congress/data}"
DATA_DIR="${DATA_DIR:-/app/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/update-$DATE.log") 2>&1

echo "========================================================"
echo "  CSearch NLP nightly update — Congress $CONGRESS"
echo "  $(date)"
echo "========================================================"

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Fetch new bill text from GovInfo
# Fetcher skips bills where .meta.json already exists, so only new bills
# discovered by the scraper since the last run are downloaded.
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/4] Fetching new bill text from GovInfo ---"
python fetcher.py \
  --data-root "$CONGRESS_DATA_ROOT" \
  --congresses "$CONGRESS" \
  --workers 16

# ---------------------------------------------------------------------------
# Step 2: Content hash check
# Compares SHA256 hashes of bill text content (XML attributes stripped) against
# the stored manifest from the previous run. Bills where only metadata changed
# (e.g. action dates in XML attributes) are treated as unchanged.
# Exits 0 if any bills changed, 1 if nothing changed.
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/4] Checking for content changes ---"
if ! python content_hasher.py \
     --congress "$CONGRESS" \
     --data-dir "$DATA_DIR"; then
  echo "No meaningful content changes detected — skipping pipeline."
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Chunk
# Rechunks the full congress (fast, CPU-only, ~minutes).
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/4] Chunking ---"
python chunker.py --congresses "$CONGRESS"

# ---------------------------------------------------------------------------
# Step 4: Embed
# Incremental — chunks whose identity (bill_id + content hash + chunk_index)
# already exists in the output shard are skipped. Only new chunks cost money.
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/4] Embedding new chunks ---"
python embedder.py \
  --input "$DATA_DIR/processed_chunks/$CONGRESS" \
  --output "$DATA_DIR/embedded_chunks/$CONGRESS"

# ---------------------------------------------------------------------------
# Step 5: Upsert
# Idempotent per bill_id. --skip-hnsw lets pgvector update the HNSW index
# incrementally on insert. Run with --index-only monthly for a full rebuild.
# ---------------------------------------------------------------------------
echo ""
echo "--- [5/5] Upserting into PostgreSQL ---"
python upserter.py \
  --input "$DATA_DIR/embedded_chunks/$CONGRESS" \
  --skip-hnsw \
  --batch-size 2000

echo ""
echo "========================================================"
echo "  Done — $(date)"
echo "========================================================"
