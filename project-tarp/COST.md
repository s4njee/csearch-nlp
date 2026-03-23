# Project TARP — Cost Notes

This file summarizes the expected OpenAI API costs for the Project TARP proof of concept.

Pricing was checked against OpenAI’s official pricing pages on March 23, 2026:

- [OpenAI API pricing](https://platform.openai.com/pricing)
- [text-embedding-3-small model page](https://developers.openai.com/api/docs/models/text-embedding-3-small)

## 1. Embedding The Whole Congress 110 Bill Corpus

The current Project TARP chunking run produced:

- `439,890` chunks
- `55,260,842` embedding tokens

Using `text-embedding-3-small` at `$0.02 / 1M tokens`:

```text
55,260,842 / 1,000,000 * $0.02 = $1.1052
```

So the full embedding cost for the current Congress 110 corpus is about:

- **$1.11**

If you used the Batch API price for embeddings (`$0.01 / 1M tokens`), the same run would be about:

- **$0.55**

## 2. Cost Of Generating A Query Embedding

Query embeddings also use `text-embedding-3-small` at `$0.02 / 1M tokens`.

Typical user queries are tiny, often around `10–30` tokens.

Examples:

- `10` tokens: `10 / 1,000,000 * $0.02 = $0.0000002`
- `20` tokens: `20 / 1,000,000 * $0.02 = $0.0000004`
- `50` tokens: `50 / 1,000,000 * $0.02 = $0.000001`

So a single query embedding is effectively:

- **well under one-thousandth of a cent**
- roughly **$0.0000002 to $0.000001** for normal query lengths

## 3. GPT-5.4-Nano Answer Generation Cost

The current `query.py` uses `gpt-5.4-nano` for answer generation.

Important note:

- OpenAI’s public pricing page currently lists **`gpt-5-nano`**
- I did **not** find a distinct public pricing line item for **`gpt-5.4-nano`**
- So the numbers below use **`gpt-5-nano` pricing as an inference**

Current official `gpt-5-nano` pricing:

- **Input:** `$0.05 / 1M tokens`
- **Output:** `$0.40 / 1M tokens`

## 4. Estimated Per-Query Answer Cost

A typical RAG query here looks something like:

- query text: very small
- top 5 retrieved chunks
- chunk snippets passed into the answer model

If the answer prompt is about `3,000` input tokens and the model returns `500` output tokens:

```text
Input:  3,000 / 1,000,000 * $0.05  = $0.00015
Output:   500 / 1,000,000 * $0.40  = $0.00020
Total:                                 $0.00035
```

So a normal answer-generation call is on the order of:

- **$0.0002 to $0.001 per query**

That means the answer-generation cost is still extremely small for this POC.

## 5. Practical Takeaways

- Embedding the full current Congress 110 corpus is cheap: about **$1.11**
- Query embeddings are basically free
- Answer generation with `gpt-5-nano`-class pricing is also very cheap per query
- The one-time corpus embedding cost is the main OpenAI spend for this POC

## 6. Summary Table

| Item | Basis | Estimated Cost |
|---|---|---:|
| Full Congress 110 embedding | 55,260,842 tokens @ `$0.02 / 1M` | **$1.1052** |
| Full Congress 110 embedding via Batch API | 55,260,842 tokens @ `$0.01 / 1M` | **$0.5526** |
| One query embedding | 10–50 tokens @ `$0.02 / 1M` | **$0.0000002–$0.000001** |
| One answer generation call | ~3K input + 500 output using `gpt-5-nano` pricing | **~$0.00035** |

## 7. Caveat

OpenAI pricing can change. Re-check the official pricing page before doing a larger full-history embedding run.
