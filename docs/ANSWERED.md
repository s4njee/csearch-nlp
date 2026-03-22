# Archived Open Questions — Resolved

These questions were originally listed in the `README.md` Open Questions section. They have been answered and the decisions are now reflected in the main documentation.

---

**Q1 — Embedding model consistency across dev and prod.**
If dev uses Ollama `nomic-embed-text` (768 dims) and prod uses OpenAI `text-embedding-3-small` (1536 dims), the Qdrant collections are structurally incompatible — you'd have to re-embed the entire corpus when migrating to prod. Options: (a) use OpenAI for both dev and prod (accept the ~$32 batch cost), (b) use Ollama for both and accept slightly lower embedding quality in prod, or (c) treat the dev collection as throwaway and plan a full re-embed for prod. Which approach?

> **A1:** We will use OpenAI on Congress 110 for dev (Project TARP). When doing the full Congress range, we will re-evaluate Ollama vs. OpenAI at that time.

---

**Q2 — Ollama networking in K8s.**
Ollama runs on the local machine with the RTX 3090. Is it a native host process or containerized? The K8s pods (API and sync CronJob) need a reachable endpoint — either `http://host.docker.internal:11434`, a K8s Service pointing to a host port, or a NodePort. What's the current Ollama setup?

> **A2:** It was running on bare metal, but for now the focus is on OpenAI. Ollama networking is deferred until/if local inference is needed again.

---

**Q3 — LLM provider for generation.**
Three options appear across the docs: Claude Sonnet (NLP.md), OpenAI (README), and Ollama `qwen2.5:7b` (IMPLEMENTATION.md configmap). For dev, Ollama is free but lower quality. For prod, Claude or OpenAI gives much better answer synthesis. Is the plan to use Ollama for dev iteration and Claude Sonnet for prod? Should the code abstract over both via a common interface?

> **A3:** The plan is OpenAI for Project TARP. Specifically `text-embedding-3-small` for embeddings and `gpt-5.4-nano` for answer generation.

---

**Q4 — Git hosting and ArgoCD.**
ArgoCD needs a reachable git remote to sync from. Is the `csearch-nlp` repo going on GitHub/GitLab, or is there a self-hosted Gitea/Forgejo instance on the cluster? If it's not yet on a remote, ArgoCD can't auto-sync — deployment would be manual `kubectl apply` until the repo is pushed upstream.

> **A4:** We will set up a GitHub repo for ArgoCD to sync from.

---

**Q5 — Storage backend: hostPath vs. NFS.**
`IMPLEMENTATION.md` uses `hostPath` PVs (fast, local SSD). `NLP.md` uses NFS PVs (`nfs-client` on `10.0.0.3`). For Qdrant with on-disk HNSW, SSD latency matters — NFS adds a network hop that could push cold-cache queries from ~150ms to 300ms+. Is the `mars` dev node equipped with local SSD? Should we use `hostPath` for `mars` and defer the NFS decision to `netcup` prod?

> **A5:** hostPath for `mars`. Will likely use hostPath for prod too.

---

**Q6 — Production migration strategy.**
The prod environment (`netcup`) is a separate remote VPS without a GPU. When it's time to deploy there, the Qdrant DB either needs to be (a) re-embedded from scratch using API-based embeddings, (b) snapshot-synced from `mars`, or (c) served from `mars` and proxied. How much thought should go into this now vs. after the dev pipeline stabilizes?

> **A6:** This will likely tip the hat towards OpenAI for prod. We are using OpenAI for Project TARP anyways, so this decision is deferred.

---

**Q7 — Vote data.**
The README and Phase 1 CLI commands reference `--include-votes` and a `vote_chunks` collection, but `NLP.md` only defines a `bill_chunks` collection. Are votes a separate collection, a chunk type within `bill_chunks`, or deferred to a later phase?

> **A7:** We need a `vote_chunks` collection — votes will be a separate Qdrant collection, not mixed into `bill_chunks`.

---

**Q8 — API authentication.**
For `mars` dev, no auth is needed. But once this is exposed via the frontend in production, should the `/api/nlp/search` endpoint require authentication? Rate limiting? The LLM generation step is the most expensive per-request — an unauthenticated endpoint could run up costs quickly.

> **A8:** No authentication for now. Will be revisited before production exposure.

---

**Q9 — Cross-encoder model size on VPS.**
The cross-encoder (`ms-marco-MiniLM-L-12-v2`, ~130MB on disk, ~500MB under load) runs on CPU within the API pod. With the API pod limited to 2 GB, and the model taking ~500 MB, that leaves ~1.5 GB for the FastAPI process, async request handling, and Python overhead. Is this sufficient under concurrent load, or should we consider a lighter model or a separate sidecar?

> **A9:** This won't be needed if we end up using OpenAI. Deferred to later.
