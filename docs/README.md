# Talent Radar — Contributor Documentation

Welcome! This folder is the technical handbook for people who want to read,
run, debug or extend Talent Radar. It describes the system **as the code
behaves today**; where the top-level `README.md` or
`redrob_ranking_architecture.md` differ from the code, the code is the source
of truth and the difference is listed in [known-issues-and-roadmap.md](known-issues-and-roadmap.md).

## What the project does

Talent Radar ranks a pool of ~100,000 candidate profiles against one job
description (Redrob's *Senior AI Engineer — Founding Team* JD) and emits the
**top 100** as `submission.csv` (`candidate_id, rank, score, reasoning`).

The hard parts, and where each is solved:

| Problem | Where it is handled | Doc |
|---|---|---|
| 100K profiles is too many to score carefully in 5 CPU-minutes | Hybrid retrieval (FAISS ∪ BM25) → ~3K shortlist | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Keyword-stuffed profiles (e.g. "HR Manager" with 9 AI skills) | Title/role-coherence features + learned ranker | [retrieval-and-ranking](retrieval-and-ranking.md) |
| ~80 deliberately impossible "honeypot" profiles | OR-logic implausibility rules, hard-filtered | [honeypot-detection](honeypot-detection.md) |
| JD has explicit "do not want" clauses | Rule features fed to a Ridge ranker | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Candidates who look perfect but are unreachable | Behavioural score from 23 platform signals | [retrieval-and-ranking](retrieval-and-ranking.md) |
| Every finalist needs a grounded, non-repetitive explanation | Reason planner + local Qwen + template fallback | [reasoning-engine](reasoning-engine.md) |
| No ground-truth relevance labels | Synthetic stratified eval set + CV model selection | [evaluation-and-testing](evaluation-and-testing.md) |

## Reading order

1. [getting-started.md](getting-started.md) — set up, run, what you can and cannot run from a fresh clone.
2. [architecture.md](architecture.md) — the offline/online split and the 10-stage ranking run.
3. [data-model.md](data-model.md) — input schema, `features.parquet` columns, every artifact.
4. [retrieval-and-ranking.md](retrieval-and-ranking.md) — embeddings, BM25, features, behavioural score, Ridge vs LightGBM.
5. [honeypot-detection.md](honeypot-detection.md) — the six rules and the company-timeline statistics.
6. [reasoning-engine.md](reasoning-engine.md) — `CandidateAnalysis → ReasonPlan → LLM / fallback`.
7. [evaluation-and-testing.md](evaluation-and-testing.md) — eval-set construction, metrics, validators, tests.
8. [app-and-deployment.md](app-and-deployment.md) — Streamlit app, Hugging Face Space, CI sync.
9. [design-decisions.md](design-decisions.md) — why things are the way they are, with alternatives and trade-offs.
10. [known-issues-and-roadmap.md](known-issues-and-roadmap.md) — honest list of gaps, doc/code drift, and good first improvements.
11. [faq.md](faq.md) — deeper "why / what if" questions about the design.
12. [glossary.md](glossary.md) — domain and project vocabulary.

## Repository map

```
.
├── rank.py                 # THE online entry point (graded, ≤5 min, CPU only)
├── reasoning.py            # Reason planner + Qwen generation + template fallback
├── features_utils.py       # JD-rule features, behavioural score, notice-period penalty
├── schema_utils.py         # Shared constants/helpers (consulting firms, date parsing)
│
├── parse_candidates.py     # OFFLINE 1: JSONL(.gz) → artifacts/features.parquet
├── company_stats.py        # OFFLINE 2: per-company "earliest plausible start" statistics
├── honeypot_flags.py       # OFFLINE 3: six OR-logic implausibility rules
├── build_index.py          # OFFLINE 4: MiniLM embeddings → FAISS, BM25, JD embeddings
├── label_eval_set.py       # OFFLINE 5: stratified 150-row eval set
├── train_ranker.py         # OFFLINE 6: Ridge vs LightGBM bake-off → ranker_model.pkl
│
├── validate_local.py       # Superset of the official validator (+ reasoning checks)
├── validate_submission.py  # Official format validator (as supplied)
├── tests/                  # pytest: no-stuffing regression, Streamlit data loading
├── app/app.py              # Streamlit: Pipeline Runner + Candidate Explorer
├── artifacts/              # Generated/precomputed data (large files via Git LFS or gitignored)
├── job_description.docx    # The JD (parsed at runtime for BM25 + embeddings)
├── submission.csv          # Latest produced ranking
├── submission_metadata.yaml
├── redrob_ranking_architecture.md   # Original design spec (partly superseded; see known issues)
└── .github/workflows/sync_to_hf.yml # Push main → Hugging Face Space
```

> **Note on layout.** The top-level README and the design spec refer to a
> `precompute/` directory. In the repository the precompute scripts live at
> the repo root next to `rank.py`. All commands in these docs use the real paths.

## Conventions used in this codebase

- Python ≥ 3.13 (see `submission_metadata.yaml` / HF Space config), `from __future__ import annotations` everywhere.
- Every function carries a one-line `# comment` above it; module docstrings use `@file` / `@package` tags (`precompute` or `online_ranking`).
- Scripts are CLI-first (`argparse`) with sensible defaults pointing at `artifacts/`.
- Nested JSON (career, skills, education) is stored as **JSON strings** inside parquet columns and re-parsed on demand — see [data-model.md](data-model.md).
- Determinism matters: seeds are fixed (`random_state=42/123`), tie-breaks are explicit, behaviour scoring uses a pinned "today" (`2026-06-30`).
