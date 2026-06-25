# Retrieval Evaluation

Measure semantic-search quality so changes to chunking, the embedding model,
HNSW settings, or rank fusion can be judged as improvements or regressions
instead of guessed at. Productize retrieval *before* productizing generated
answers.

_Last verified against code: 2026-05-30._

## Files

- `eval_set.json` — labeled queries in three categories with expected bill ids.
- `run_eval.py` — harness that runs the set against `nlp.*` and reports metrics.

## Target eval set

Curate toward this mix (the checked-in file is a starter seed):

| Category | Count | Examples |
| --- | --- | --- |
| Policy / natural language | 50 | "lowering prescription drug costs for seniors" |
| Exact bill number / title | 25 | "HR 42", "Inflation Reduction Act" |
| Sponsor / committee / procedure | 25 | "bills sponsored by Padilla", "motion to recommit" |

For each query record `expected_bill_ids` (relevant) and optional
`unacceptable_bill_ids` (clear false positives).

## Metrics

The harness reports, per category and overall:

- **recall@k** — fraction of expected bills found in the top k
- **precision@k** — fraction of the top k that are relevant
- **MRR** — mean reciprocal rank of the first relevant bill
- **latency p50/p95** — wall-clock per query

Track OpenAI embedding **cost per 1,000 searches** separately from the
`/metrics` `csearch_semantic_total` counter once the query-embedding cache is
warm.

## Running

```bash
# Offline smoke against fixtures (no OpenAI key needed)
docker compose up -d postgres            # or any pgvector Postgres
python db/migrate.py --dsn "$DSN"
psql "$DSN" -f db/seed/fixtures.sql
uv run backend/nlp/eval/run_eval.py --provider fake --only-fixtures --mode vector --dsn "$DSN"

# Real evaluation against production-shaped data
OPENAI_API_KEY=sk-... uv run backend/nlp/eval/run_eval.py --mode vector
OPENAI_API_KEY=sk-... uv run backend/nlp/eval/run_eval.py --mode hybrid   # keyword+vector RRF
```

Compare `--mode vector` against `--mode hybrid` on the same set: adopt the
hybrid path in the production API (`csearch_api.hybrid.fuse_ids`) only if it
wins on recall/MRR without a latency regression.

## Treating model changes as migrations

A new embedding model is a data migration, not a config flip:

1. add a new embedding table/column partition,
2. backfill,
3. run this eval against both,
4. switch traffic,
5. keep the old table for rollback.
