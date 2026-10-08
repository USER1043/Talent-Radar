# Talent Radar — Contributor Documentation

Welcome! This folder is the technical handbook for people who want to read,
run, debug or extend Talent Radar. It describes the system **as the code
behaves today**; where the top-level `README.md` or
`redrob_ranking_architecture.md` differ from the code, the code is the source
of truth and the difference is listed in [known-issues-and-roadmap.md](known-issues-and-roadmap.md).

## What the project does

Talent Radar ranks a pool of ~100,000 candidate profiles against one job
description (Redrob's *Senior AI Engineer — Founding Team* JD) and emits the
**top 100** as `submission.csv` (`candidate_id, rank, score, reasoning`). The ranking step is CPU-only, makes no network or LLM-API calls
by default, and takes about 1 min 47 s (4 GB) on 16 cores.

The hard parts, and where each is solved:

| Problem | Where it is handled | Doc |
|---|---|---|
| 100K profiles is too many to score carefully in 5 CPU-minutes | Hybrid retrieval (FAISS ∪ BM25) → ~3K shortlist | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Keyword-stuffed profiles (e.g. "HR Manager" with 9 AI skills) | Title/role-coherence features + learned ranker | [retrieval-and-ranking](retrieval-and-ranking.md) |
| ~80 deliberately impossible "honeypot" profiles | Five self-contained OR-logic rules, hard-filtered | [honeypot-detection](honeypot-detection.md) |
| JD has explicit "do not want" clauses | Rule features fed to a Ridge ranker | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Candidates who look perfect but are unreachable | Behavioural score + post-model availability multiplier | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Every finalist needs a grounded, non-repetitive explanation | Fact-grounded generator with a claim audit; optional offline, gated LLM rewrites | [reasoning-engine](reasoning-engine.md) |
| No ground-truth relevance labels | Synthetic stratified eval set + CV model selection | [evaluation-and-testing](evaluation-and-testing.md) |

## Status at a glance

A snapshot of where the project stands at the time these docs were written (figures marked † are quoted from the project's own README/architecture notes, not re-measured).

**Working well**
- Full 100K run in **1 min 47 s, 4.0 GB**, CPU-only, no network or LLM at ranking time †.
- Hybrid retrieval (FAISS ∪ BM25) → 2,969-candidate shortlist †; deterministic tie-breaks; output passes the organisers' validator †.
- Reasoning is fact-grounded and audited: 0 failures over ≈ 800 claims checked against the raw data †; LLM polish is offline, gated and cached.
- Honeypot filter uses only self-contained rules; 0 flagged candidates in the top 100 †. A harmful rule (company founding) was found by audit and removed.
- Availability is handled explicitly (multiplier) and a wrong-signed notice coefficient is neutralised.
- Reproducibility: fixed seeds, pinned dependencies, scripted precompute, tests for the reasoning stack (29 pass †).

**Weak or open**
- **No ground truth.** Eval labels are rule-generated; the ranker's CV score measures agreement with those rules, not the hidden relevance tiers.
- **Three JD disqualifiers have ≈ 0 weight** (consulting-only, pure research, wrapper-only); the committed eval set has only 4 of 150 consulting-only rows.
- **Honeypot filter is broad** (5.5% of the pool flagged vs ≈ 80 expected †, mostly via the education-order rule) and its real recall is unknown.
- Eval set and ranker were **not regenerated** after the honeypot rule change.
- Default reasoning is template-based (phrasing repeats); LLM rewrites cover only ≈ 38% of rows †; slotted mode is unfinished.
- `rank.py` index/row alignment can break for mid-sized subset inputs; dashboard KPIs are hard-coded and stale; CI runs no tests.
- Committed `jd_embeddings.npy` cannot be reproduced by `build_index.py` (max abs diff 0.095 †).

Details and a prioritised fix list: [known-issues-and-roadmap.md](known-issues-and-roadmap.md).

## Reading order

1. [getting-started.md](getting-started.md) — set up, run, what you can and cannot run from a fresh clone.
2. [architecture.md](architecture.md) — the offline/online split and the 10-stage ranking run.
3. [data-model.md](data-model.md) — input schema, `features.parquet` columns, every artifact.
4. [retrieval-and-ranking.md](retrieval-and-ranking.md) — embeddings, BM25, features, behavioural score, Ridge coefficients, availability multiplier.
5. [honeypot-detection.md](honeypot-detection.md) — the five rules, measured flag rates, and the removed company-timeline rule.
6. [reasoning-engine.md](reasoning-engine.md) — grounded generator + audit, offline LLM-rewrite cache, slotted templates, legacy planner.
7. [evaluation-and-testing.md](evaluation-and-testing.md) — eval-set construction, metrics, validators, tests.
8. [app-and-deployment.md](app-and-deployment.md) — Streamlit app, Hugging Face Space, CI sync.
9. [design-decisions.md](design-decisions.md) — why things are the way they are, with alternatives and trade-offs.
10. [known-issues-and-roadmap.md](known-issues-and-roadmap.md) — honest list of gaps, doc/code drift, and good first improvements.
11. [faq.md](faq.md) — deeper "why / what if" questions about the design.
12. [glossary.md](glossary.md) — domain and project vocabulary.

## Repository map

```
.
├── rank.py                 # THE online entry point (graded, ≤5 min, CPU only); --reasoning-mode selects the explanation generator
├── features_utils.py       # JD-rule features, behavioural score, notice penalty, availability multiplier, AS_OF_DATE
├── schema_utils.py         # Shared constants/helpers (consulting firms, date parsing)
├── reasoning_grounded.py   # DEFAULT reasoning: facts from the record + claim audit
├── reasoning_llm_rewrite.py# Optional offline LLM polish: gate + cache (grounded-llm / grounded-cached)
├── reasoning_slotted.py    # Slot-filled LLM templates (slotted-cached; build path unfinished)
├── reasoning.py            # LEGACY planner + Qwen generation (--reasoning-mode llm)
│
├── parse_candidates.py     # OFFLINE 1: JSONL(.gz) → artifacts/features.parquet
├── honeypot_flags.py       # OFFLINE 2: five OR-logic implausibility rules
├── build_index.py          # OFFLINE 3: MiniLM embeddings → FAISS, BM25, JD embeddings
├── label_eval_set.py       # OFFLINE 4: stratified 150-row eval set (rule-generated labels)
├── train_ranker.py         # OFFLINE 5: Ridge vs LightGBM bake-off → ranker_model.pkl
├── company_stats.py        # RETIRED: fed the removed honeypot rule 3; kept for reference
│
├── validate_local.py       # Superset of the official validator (+ ID existence, reasoning checks)
├── validate_submission.py  # Official format validator (as supplied)
├── tests/                  # pytest: reasoning (grounded / llm-rewrite / slotted), no-stuffing regression, Streamlit data loading
├── app/app.py              # Streamlit: Pipeline Runner + Candidate Explorer
├── artifacts/              # features/honeypot parquet (LFS), eval_set.csv, ranker_model.pkl, jd_embeddings.npy, reasoning_rewrites.json
│                           #   (faiss.index and bm25.pkl are generated and git-ignored)
├── job_description.docx    # The JD (parsed at runtime for BM25 + embeddings)
├── submission.csv          # Latest produced ranking
├── submission_metadata.yaml
├── redrob_ranking_architecture.md   # v2.0 "as built" design notes (rewritten after an audit)
└── .github/workflows/sync_to_hf.yml # Push main → Hugging Face Space
```

> **Note on layout.** The layout is flat — there is no `precompute/` directory; precompute scripts live at the repo root next to `rank.py`
> (older material, including parts of `.gitignore`, still mentions `precompute/`).

## Conventions used in this codebase

- Python ≥ 3.13 (see `submission_metadata.yaml` / HF Space config), `from __future__ import annotations` everywhere.
- Every function carries a one-line `# comment` above it; module docstrings use `@file` / `@package` tags (`precompute` or `online_ranking`).
- Scripts are CLI-first (`argparse`) with sensible defaults pointing at `artifacts/`.
- Nested JSON (career, skills, education) is stored as **JSON strings** inside parquet columns and re-parsed on demand — see [data-model.md](data-model.md).
- Determinism matters: seeds are fixed (`random_state=42/123`), tie-breaks are explicit, recency uses a pinned `AS_OF_DATE` (`2026-06-30`), default reasoning is a pure function of record + rank.
- Never trust generated text without checking it against the record: new reasoning code must pass through `audit_reasoning` (or a gate that calls it).
