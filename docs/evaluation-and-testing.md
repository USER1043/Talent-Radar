# Evaluation & Testing

## The label problem

The challenge supplies no relevance labels, yet the ranker needs supervision and every submission is capped at three attempts. The project's
answer is a **small, stratified, reproducible eval set** plus cross-validated model selection.

## Eval-set construction (`label_eval_set.py`)

Output: `artifacts/eval_set.csv` — **150 rows**, shuffled (`random_state=123`). Sampling is seeded (`42`).
It needs `features.parquet`, `honeypot_flags.parquet`, `faiss.index`, `jd_embeddings.npy`.

Similarity for each candidate is looked up from a FAISS search of the **top 5,000** vs. the ideal-candidate vector; everyone outside that
set gets `semantic_sim = 0.0`.

| `sample_type` | n | Pool definition | `relevance` |
|---|---|---|---|
| `honeypot` | 30 | `is_honeypot == True` | 0 |
| `keyword_stuffer` | 40 | not honeypot, **non-ML title**, `semantic_sim > 0.4` | 1 |
| `strong_fit` | 40 | not honeypot, **ML title**, not consulting-only, `semantic_sim > 0.4` | 5 |
| `mild_fit` | 20 | not honeypot, ML title, and (consulting-only **or** `semantic_sim ≤ 0.4`) | 3 |
| `background` | 20 | random remainder | 1 if ML title else 0 |

`is_ml_title` = title contains one of *ml, machine learning, ai, artificial intelligence, nlp, computer vision, data scientist,
applied scientist* and none of *marketing, hr, recruiter, sales, support, operations*.

Resulting label counts: 0 → 50, 1 → 40, 3 → 20, 5 → 40.

> **Important:** relevance is assigned *by these rules*, not by a human reading each profile (the script's docstring says
> "automatically assigns baseline relevance scores"; the original plan envisaged manual labelling, and the project README now states this plainly). Relevance is constant within each
> sample type in the committed file. The CSV has human-scannable columns so labels *can* be edited by hand and the ranker retrained.
> The 30 honeypot negatives were sampled from flags that **still included the since-removed rule 3**, and the eval set / ranker have not been regenerated since.
> Consequences are discussed in [known-issues-and-roadmap.md](known-issues-and-roadmap.md).

## Metrics

Implemented in `train_ranker.py`:

| Metric | Definition |
|---|---|
| NDCG@k | `Σ rel_i / log2(i+2)` over the first *k*, divided by the same with relevances sorted descending |
| MAP | Average precision with `relevance ≥ 3` as relevant |
| P@10 | Fraction of the first 10 with `relevance ≥ 3` |
| Composite | `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` |

Cross-validation: `KFold(n_splits=5, shuffle=True, random_state=42)`. In each fold the held-out rows are sorted by predicted score
and the metrics computed on that single ranked list. With ~30 rows per fold, NDCG@50 is effectively NDCG over the whole fold.
These numbers are a **sanity check on ordering**, not an estimate of the hidden grader's score.

`train_ranker.py` prints both models' mean CV composite, chooses the winner, and prints feature importances (LightGBM) or signed coefficients (Ridge).

## Validators

### `validate_submission.py <file>.csv` (official)

Exact header; exactly 100 data rows; `CAND_xxxxxxx` IDs; no duplicate IDs; integer ranks 1–100 each exactly once; float scores;
non-increasing by rank; ties broken by `candidate_id` ascending. Reads the CSV only — it never checks that IDs exist in the pool.

### `validate_local.py --csv submission.csv [--candidates candidates.jsonl]`

A superset. Adds:

- **ID existence** against the candidates file (the gap the official validator leaves).
- **Non-empty** reasoning and **no duplicate** reasoning strings (errors).
- **Tone warning**: any row with rank > 70 whose reasoning contains none of a list of hedging words (*gap, concern, limited, lack, notice, however, but, …*). The grounded reasoning states concerns with other wording
  ("Caveat:", "Held back by"), so this warning can be a **false positive**.

Exit code 1 on errors, 0 otherwise; warnings print but don't fail.

## Tests (`pytest -q` from the repo root)

| File | Tests | Purpose | Needs |
|---|---|---|---|
| `tests/test_no_stuffing_regression.py` | `test_no_template_reasoning_pattern` | Top-10 reasoning must not match the stuffing template (`HR Manager|Content Writer … N.N yrs … N AI core skills`) | `submission.csv` |
| | `test_no_overlap_with_sample_top10` | Top-10 must not share candidates with the known-bad sample submission | `sample_submission.csv` (not committed) |
| | `test_reasoning_is_full_sentences` | Top-10 reasoning is ≥ 20 chars and ends in `.!?` | `submission.csv` |
| | `test_no_duplicate_reasonings` | All 100 reasonings unique | `submission.csv` |
| | `test_scores_non_increasing` | Score ordering | `submission.csv` |
| `tests/test_streamlit_app.py` | `test_streamlit_load_data` | `load_data` returns all columns the UI needs (`behavioral_score`, trap flags, …) | `features.parquet`, `honeypot_flags.parquet` (LFS), `submission.csv`; run from repo root |
| `tests/test_reasoning_grounded.py` | 8 tests | Audit passes; only owned skills named; real, specific concerns; honest no-concern case; tone follows rank; determinism; many candidates distinct and grounded; audit catches fabrication | none (builds synthetic rows) |
| `tests/test_reasoning_llm_rewrite.py` | 17 tests | The rewrite gate (what it accepts/rejects), keep-list, prefill, fallback, repair pass, cache round-trip, live-mode cache write | none (injected fake generator) |
| `tests/test_reasoning_slotted.py` | slot/gate/cache/diversity tests | Slotted-template gate and cache refill | none |

The project reports **29 passing and 1 setup error** in a fresh clone (the error is the missing `sample_submission.csv`). There are no direct unit tests yet for `features_utils.py` rules, the
availability multiplier, `honeypot_flags.py`, or `rank.py` itself.

Defined but unused constants in the regression test (`JD_TITLE_KEYWORDS`, `MISMATCH_TITLE_PATTERNS`) hint at a planned check:
"no title-mismatched candidate in the top 10". Implementing that is a good first contribution.

## Suggested pre-submission checklist

1. `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` (note wall-clock; budget is 5 min; measured 1 min 47 s on 16 cores).
2. `python validate_submission.py submission.csv`
3. `python validate_local.py --csv submission.csv --candidates ./candidates.jsonl`
4. `pytest -q`
5. Confirm no honeypot-flagged candidates in your top 100 (currently 0; limit is 10%). Remember the filter flags 5.5% of the pool, so absence in the top 100 is expected but says little about recall of the ≈80 true honeypots.
6. Skim ranks 1–10 by hand: titles, YOE, notice, company — do they look like the JD's "ideal candidate"?

## Ideas for stronger evaluation

- Hand-review and correct the 150 labels (or add a pairwise "obviously better/worse" regression set as the design doc suggests).
- Split the eval set by `sample_type` for CV so that stratification — not just random folds — is respected.
- Report per-type metrics (does the ranker still push honeypots/stuffers to the bottom?).
