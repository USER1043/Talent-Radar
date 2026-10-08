# Known Issues & Roadmap

An honest ledger of where the code, the docs and the ideal design diverge. Items are ordered roughly by impact.
Where an item says *verify*, it follows from reading the code but has not been reproduced with the full dataset.

## A. Correctness risks

### A1. Row-position coupling between artifacts and filtered data (`rank.py`) — *verify*

`faiss.index` and `bm25.pkl` are built over `features.parquet` in its original row order, so their result indices are
**positions in the full 100K table**. `rank.py` first filters `df_feat` to the IDs present in the supplied candidates file and
calls `reset_index(drop=True)`. When the supplied file is the full pool (the graded case) the order is unchanged and everything lines up.
When it is a **subset** (e.g. the Streamlit Pipeline Runner with an uploaded sample):

- `df_feat.index.isin(shortlist_indices)` compares positions from the *full* table against positions in the *filtered* table;
- `feat_id_to_idx` (used to look up `bm25_all_scores`) maps IDs to *filtered* positions, so BM25 scores can be read from the wrong candidates.

Fix: keep the original row position as a column (e.g. `df_feat["_pos"] = range(len(df_feat))` before filtering) and index FAISS/BM25 results through it.

### A2. Fewer than 100 rows possible

If the shortlist (after honeypot filtering) has < 100 rows the CSV is written with fewer rows and both validators will fail. `rank.py` neither pads nor warns.

### A3. `validate_local.validate` return type

`validate()` returns a bare list on the "could not parse CSV" path but a `(errors, warnings)` tuple otherwise; `main()` unpacks a tuple, so a parse failure raises instead of reporting.

### A4. Substring matching is loose

- `is_ml_title` (eval set) tests `"ai" in title`, which also matches e.g. "Retail", "Maintenance"; `"ml"` matches "html".
- `compute_no_recent_coding` tests `"sales" in title`, which also matches "Salesforce Developer".
- Research/production/validation term lists are plain substrings ("scale" matches "scalable", "talk" matches "talking").

Prefer word-boundary regexes or token sets, and review the eval-set labels after changing them.

## B. Modelling limitations

### B1. Eval labels are rule-derived, not human-judged

Relevance is assigned by sample type (see [evaluation-and-testing.md](evaluation-and-testing.md)). Several ranker features
(`is_consulting_only`, `semantic_sim_to_ideal`, title-based flags) overlap with the definitions used to label the rows, so the model is partly learning to
reproduce the sampling rules. The CV composite is therefore optimistic and should not be read as expected leaderboard performance.

### B2. No feature scaling for Ridge

Inputs range from 0/1 flags to BM25 scores in the tens and `n_skills`/YOE in single/double digits; with a fixed `alpha=1.0` the L2 penalty affects them unevenly and coefficients are not comparable across features. Wrap in a `StandardScaler` pipeline (and persist it in the pickle).

### B3. Folds treat the whole eval set as one query

NDCG@50 on ~30-row folds equals NDCG over the fold; LambdaRank gets a single group. There is no per-query structure to learn from.

### B4. Hand-set weights and thresholds

Behavioural weights, notice-period steps, the 0.52–0.58 semantic cut-offs in `reasoning.py`, and the 8-skill/1-year honeypot thresholds are not fit to data.
The design doc asked for the honeypot thresholds to be tuned against the real distribution.

### B5. Single composite for behaviour

The ranker sees only `behavioral_score`, not its eight components, so it cannot learn that (say) response rate matters more than GitHub activity.

### B6. Shortlist sensitivity

Retrieval uses one query per leg. A strong candidate who is distant from the JD text in both embedding and BM25 space never reaches the ranker.

## C. Honeypot rules

- **Flag rate vs expectation.** The dashboard hard-codes a shortlist of 2,956 and 2,556 "valid post-honeypot", implying ~400 shortlisted candidates (~13%) were removed, while only ~80 honeypots exist in the 100K pool. The rules (especially #5, a 2-month tolerance on `duration_months`) may be over-firing. Plot flags by rule and compare against the expected prior.
- **Tier B companies** (< 5 mentions) get no timeline check.
- **Rule 4** compares only adjacent completed roles after sorting by start date and ignores current roles.

## D. Reasoning

- 0.5B-parameter output quality: generic phrasing, occasional semantic slips ("excels in high-risk roles").
- Explanations talk about rank/notice/semantic match more than candidate-specific facts (employer, projects, skills).
- Model weights are downloaded from the Hub on first use and not revision-pinned; an offline machine without cache silently takes the template fallback.
- `validate_explanation` uses crude substring replacement; first-person marker checks like `"i "` can match inside words.
- `torch` is imported at module top-level.

## E. App / ops

- Cosmetic progress bar; hard-coded KPI values (2,956 / 2,556).
- `@st.cache_data` on `load_data(csv_path)` is keyed by path; re-running the pipeline overwrites `temp_preview_ranked.csv` but the path string is unchanged, so the Explorer can show stale rows until the cache is cleared.
- `temp_preview_ranked.csv` is written to the CWD and is shared across all sessions of a deployed Space.
- Pipeline Runner depends on gitignored artifacts (`faiss.index`, `bm25.pkl`) — it cannot work on a plain clone/Space unless they are supplied.
- CI force-pushes to the Space and runs no tests.
- Pickled artifacts execute code on load.

## F. Documentation drift (code wins)

| Source | Says | Code does |
|---|---|---|
| `README.md` | scripts in `precompute/` | scripts at repo root |
| `README.md` / spec | eval set "hand-labeled" | rule-assigned relevance (`label_eval_set.py`) |
| `README.md` | Deterministic planner "eliminates LLM CPU overhead" | Planner feeds a batched Qwen call; templates are the fallback |
| `README.md` | 5 honeypot rules | 6 rules (education sequence added) |
| `README.md` pipeline diagram | "median first-seen date" | ≈10th percentile, ≥5 mentions |
| Spec §6.2 | corroborated at ≥ 2 mentions, median | ≥ 5 mentions, 10th percentile |
| Spec §6.1 | rule 2 = skill/experience ratio | `yoe ≤ 1 and n_skills ≥ 8` |
| `README.md` | shortlist ≈ 4–5K | ≈ 3K (2 × 1500 with overlap) |
| `README.md` | "Stage 10 write CSV + self-validate" | `rank.py` writes only; run `validate_local.py` separately |
| Spec §9 | LightGBM LambdaMART | Bake-off; Ridge currently wins and is committed |
| Spec §7 | smooth notice-period curve | step function |
| `schema_utils.py` | refers to `ARCHITECTURE.md` | file is `redrob_ranking_architecture.md` |

## G. Roadmap / good first issues

**Quick wins**

1. Fix A1 (carry original row position) and add a regression test that ranks a 2K-row subset and checks IDs match the full run.
2. Make `validate_local.validate` always return `(errors, warnings)`.
3. Replace substring checks with word-boundary matching; add unit tests for each `compute_*` rule in `features_utils.py`.
4. Add a `StandardScaler` to the Ridge pipeline and report coefficients on standardised features.
5. Replace hard-coded dashboard KPIs with values returned by `rank.py` (e.g. write a small JSON sidecar).
6. Implement the title-mismatch top-10 test hinted at by unused constants in `tests/test_no_stuffing_regression.py`.
7. Add a CI job that runs `pytest` and `validate_local.py` before the HF sync.

**Medium**

8. Per-rule honeypot report (counts, overlap) and tune thresholds from distributions.
9. Pairwise "obviously better/worse" regression set for the ranker, and per-sample-type CV metrics.
10. Expose behavioural components to the ranker; fit their weights.
11. Multi-query retrieval (e.g. one query per JD requirement) and reciprocal-rank fusion for recall.
12. Vectorise `iterrows()` hot spots (`extract_candidate_features`, `honeypot_flags`).

**Larger**

13. Human-reviewed relevance labels; check agreement between rules and reviewers.
14. Cross-encoder or LLM re-ranking of the top ~50 under the time budget.
15. Reasoning that cites candidate-specific evidence (selected sentences from career history) with an automatic grounding check.
16. Cache the Qwen/MiniLM weights in the repo build or a Docker image for truly offline runs.
