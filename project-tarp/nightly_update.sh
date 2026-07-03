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
NLP_INCREMENTAL_MAX_BILLS="${NLP_INCREMENTAL_MAX_BILLS:-1000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/update-$DATE.log") 2>&1

# Sentinel cleared at the start of every run; touched only after a successful
# upsert so the orchestrator can decide whether to trigger a frontend redeploy.
DEPLOY_SENTINEL="$DATA_DIR/.deploy-pending"
rm -f "$DEPLOY_SENTINEL"

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

CHANGES_MANIFEST="$DATA_DIR/hash_manifests/$CONGRESS.changes.json"
read -r CHANGED_COUNT MISSING_EMBEDDING_COUNT < <(
  python - "$CHANGES_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

print(data.get("changed_count", 0), data.get("missing_embedding_count", 0))
PY
)

PROCESSED_ROOT="$DATA_DIR/processed_chunks"
EMBEDDED_ROOT="$DATA_DIR/embedded_chunks"
CHUNKER_EXTRA_ARGS=()

if [ "$CHANGED_COUNT" -gt 0 ] && [ "$CHANGED_COUNT" -le "$NLP_INCREMENTAL_MAX_BILLS" ]; then
  echo "Using incremental NLP path for $CHANGED_COUNT changed bill(s)."
  PROCESSED_ROOT="$DATA_DIR/processed_chunks_delta"
  EMBEDDED_ROOT="$DATA_DIR/embedded_chunks_delta"
  CHUNKER_EXTRA_ARGS=(--bill-ids-file "$CHANGES_MANIFEST")
  # Delta dirs are per-run scratch. The chunker clears its own output, but the
  # embedder does not — so wipe both for this congress, else the upserter (which
  # globs the whole dir) would re-push bills left over from a previous delta run.
  rm -rf "$PROCESSED_ROOT/$CONGRESS" "$EMBEDDED_ROOT/$CONGRESS"
elif [ "$CHANGED_COUNT" -eq 0 ] && [ "$MISSING_EMBEDDING_COUNT" -gt 0 ]; then
  echo "No new bill text changes; retrying existing processed chunks with missing embeddings."
else
  echo "Using full-congress NLP path ($CHANGED_COUNT changed bill(s), threshold $NLP_INCREMENTAL_MAX_BILLS)."
fi

# ---------------------------------------------------------------------------
# Step 3: Chunk
# Rechunks either the changed bills or the full congress for large backfills.
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/4] Chunking ---"
if [ "$CHANGED_COUNT" -eq 0 ] && [ "$MISSING_EMBEDDING_COUNT" -gt 0 ]; then
  echo "Skipping chunker; reusing $PROCESSED_ROOT/$CONGRESS."
else
  python chunker.py \
    --congresses "$CONGRESS" \
    --output-dir "$PROCESSED_ROOT" \
    "${CHUNKER_EXTRA_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Step 4: Embed
# Incremental — chunks whose identity (bill_id + content hash + chunk_index)
# already exists in the output shard are skipped. Only new chunks cost money.
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/4] Embedding new chunks ---"
python embedder.py \
  --input "$PROCESSED_ROOT/$CONGRESS" \
  --output "$EMBEDDED_ROOT/$CONGRESS"

# ---------------------------------------------------------------------------
# Step 5: Upsert
# Idempotent per bill_id. --skip-hnsw lets pgvector update the HNSW index
# incrementally on insert. Run with --index-only monthly for a full rebuild.
# ---------------------------------------------------------------------------
echo ""
echo "--- [5/5] Upserting into PostgreSQL ---"
python upserter.py \
  --input "$EMBEDDED_ROOT/$CONGRESS" \
  --skip-hnsw \
  --batch-size 2000

# Promote the hash manifest now that fetch -> chunk -> embed -> upsert all
# succeeded (set -e means we only reach this line on success). content_hasher
# writes a .pending manifest; promoting it here — instead of inside
# content_hasher — guarantees a failed run leaves the authoritative manifest
# untouched, so the same bills are retried next run rather than stranded with
# their hashes recorded but their embeddings missing.
PENDING_MANIFEST="$DATA_DIR/hash_manifests/$CONGRESS.pending.json"
AUTH_MANIFEST="$DATA_DIR/hash_manifests/$CONGRESS.json"
if [ -f "$PENDING_MANIFEST" ]; then
  mv -f "$PENDING_MANIFEST" "$AUTH_MANIFEST"
  echo "Promoted hash manifest -> $AUTH_MANIFEST"
fi

# Mark that real content was written, so the orchestrator triggers a redeploy.
touch "$DEPLOY_SENTINEL"

echo ""
echo "========================================================"
echo "  Done — $(date)"
echo "========================================================"
