# Known Issues & Roadmap

An honest ledger of where the code, the docs and the ideal design diverge. Items are ordered roughly by impact. Where an item says *verify*, it follows from reading
the code but has not been reproduced. Figures marked *(project notes)* come from the repository's own README/architecture write-up, not from re-running the pipeline.

## What the latest audit already fixed

Useful context before you start "fixing" things that were deliberately changed (details in [design-decisions.md](design-decisions.md)):

- Honeypot **rule 3** (inferred company founding) removed: it flagged 17.5% of the pool and ejected 7 of the top 10.
- **Availability multiplier** added after the Ridge; the Ridge's wrong-signed **notice** coefficient is neutralised at prediction time.
- Default **reasoning** is now deterministic, fact-grounded and audited (0 failures over ≈ 800 claims checked against the raw file *(project notes)*); LLM polish moved offline behind a gate and cache.
- **Small-sample mode** lets the sandbox run without `faiss.index`/`bm25.pkl`.
- Project README and `redrob_ranking_architecture.md` were rewritten to say plainly that labels are rule-generated and several design points changed.

## What was independently re-checked

After `git lfs pull`, in a CPU sandbox without access to `huggingface.co` (so MiniLM could not be downloaded and `rank.py` could not run):

| Claim (project notes) | Re-checked? | Result |
|---|---|---|
| 100,000 rows, 48 columns in `features.parquet` | ✓ | Matches |
| 5,542 honeypots flagged (5.5%); rule counts 21 / 295 / 0 / 0 / 5,248 | ✓ | Matches exactly (counted from `honeypot_reasons`) |
| Ridge coefficients (e.g. +2.397 `semantic_sim_to_ideal`, +0.160 `notice_period_penalty`, ≈ 0 for the three disqualifiers) | ✓ | Match exactly (unpickled with scikit-learn 1.9.1; a version-mismatch warning vs 1.9.0, harmless here) |
| Eval-set label counts (0:50, 1:40, 3:20, 5:40) | ✓ | Match |
| Rewrite cache: 39 entries, 32 × Qwen-1.5B + 7 × Qwen-0.5B | ✓ | Match |
| "29 tests pass, 1 setup error" | ✓ | **44 pass, 1 error** in a fresh clone (more tests were added with the slotted module); the error is the missing `sample_submission.csv` |
| `is_pure_research` / `is_recent_wrapper_only` rarity | new finding | 0% / 0.18% of a 20,000-row pool sample |
| 1 min 47 s / 4 GB run, 2,969 shortlist, 2,792 post-honeypot, 0 honeypots in top 100, 0 audit failures over ≈ 800 claims, 38 rows reworded | ✗ not reproduced | Need the 100K `candidates.jsonl`, `faiss.index`/`bm25.pkl` (~37 min to build) and the MiniLM weights |

## A. Correctness risks

### A1. Row-position coupling between artifacts and filtered data (`rank.py`) — *partly mitigated, verify the rest*

`faiss.index` and `bm25.pkl` are built over `features.parquet` in its original row order, so their result indices are **positions in the full 100K table**. `rank.py` filters `df_feat` to the
IDs present in the supplied candidates file and calls `reset_index(drop=True)`.

- Full pool (the graded case): order unchanged ⇒ correct.
- **≤ 1,500 IDs**: retrieval is now skipped and `bm25_score` is set to 0 ⇒ no misalignment (at the cost of a missing BM25 feature).
- **Between 1,501 and ~100K IDs** (a large subset): FAISS/BM25 positions refer to the full table but are compared against positions in the *filtered* table, and the `bm25_all_scores` lookup uses filtered positions ⇒ wrong candidates can be shortlisted and BM25 scores read from the wrong rows.

Fix: carry the original row position as a column (`df_feat["_pos"] = range(len(df_feat))` before filtering) and index FAISS/BM25 results through it; add a regression test that a 2K-row subset ranks the same candidates as the full run restricted to them.

### A2. Fewer than 100 rows possible

If the shortlist (after honeypot filtering) has < 100 rows the CSV is written with fewer rows and both validators fail. `rank.py` neither pads nor warns. Small-sample uploads (e.g. 20 candidates) therefore produce a short, "invalid" CSV by design — fine for the sandbox, worth documenting in the UI.

### A3. Pool-only ranking

IDs not present in the precomputed `features.parquet` are silently ignored; if none match, an empty CSV is written. A sandbox sample must be drawn from the 100K pool.

### A4. `validate_local.validate` return type

`validate()` returns a bare list on the "could not parse CSV" path but a `(errors, warnings)` tuple otherwise; `main()` unpacks a tuple, so a parse failure raises instead of reporting.

### A5. Substring matching is loose

- `is_ml_title` (eval-set construction) tests `"ai" in title`, which also matches e.g. "Retail", "Maintenance"; `"ml"` matches "html".
- `compute_no_recent_coding` tests `"sales" in title`, which also matches "Salesforce Developer".
- Research/production/validation term lists are plain substrings ("scale" matches "scalable", "talk" matches "talking").

Prefer word-boundary regexes or token sets, and review the eval-set labels after changing them.

## B. Modelling limitations

### B1. No ground truth; labels are rule-derived

Relevance is assigned by sample type (see [evaluation-and-testing.md](evaluation-and-testing.md)). Several ranker features (`is_consulting_only`, `semantic_sim_to_ideal`, title-based flags) overlap with
the definitions used to label the rows, so the Ridge learns to reproduce the sampling rules. The CV score measures agreement with those rules, not with the hidden relevance tiers, and ranking quality against the hidden tiers is **unmeasured**.
The 30 honeypot negatives came from flags that included the removed rule 3 and the eval set/ranker have not been regenerated.

### B2. Three JD disqualifiers carry ≈ 0 weight

`is_consulting_only` (−0.003), `is_pure_research` (0.000) and `is_recent_wrapper_only` (0.000) are effectively not enforced by the model. Re-checked after `git lfs pull` (see "What was independently re-checked" below): `is_pure_research` fires on **0 of 20,000** randomly sampled pool candidates, so it is a *dead feature* — its rule (a research term and no production term in the narrative) essentially never matches; `is_recent_wrapper_only` fires on 0.18%; `is_consulting_only` on 7.05% of the pool. Evidence from the committed `eval_set.csv`: only **4 of the 150 rows** have `is_consulting_only = True` (3 keyword stuffers, 1 honeypot) and **none** of the 20 `mild_fit` rows do, even though that group was meant to include consulting-only candidates — the model has almost nothing to learn a weight from. (`is_pure_research` / `is_recent_wrapper_only` could not be checked here because `features.parquet` is an LFS object; they are likely also rare, and the labels come from title/similarity rules that do not use them.) The JD calls pure research a hard "we will not move forward", yet none of the three flags changes a score materially. Other coefficients have unintuitive signs (`semantic_sim_to_jd` −0.865, `years_of_experience` −0.170). Options: enforce them as an explicit multiplier (as availability is), retrain with better labels, or use monotonic constraints.

### B3. No feature scaling for Ridge

Inputs range from 0/1 flags to BM25 scores in the tens; with fixed `alpha=1.0` the L2 penalty affects them unevenly and coefficients are not comparable. Wrap in a `StandardScaler` pipeline (and persist it in the pickle).

### B4. Folds treat the whole eval set as one query

NDCG@50 on ~30-row folds equals NDCG over the fold; LambdaRank gets a single group. There is no per-query structure to learn from.

### B5. Hand-set weights and thresholds

Behavioural weights, notice steps, the availability multiplier's parameters (judgment values from the JD's wording), and the honeypot thresholds (8 skills / 1 year) are not fit to data.

### B6. Tightly packed scores

In the top 100, ranks 50–100 span only ≈ 0.4 points *(project notes)*, so small modifier changes move mid-list candidates a long way.

### B7. Single composite for behaviour; shortlist sensitivity

The ranker sees only `behavioral_score`, not its components. Retrieval uses one query per leg; a strong candidate distant from the JD text in both embedding and BM25 space never reaches the ranker.

### B8. JD embeddings are not reproducible from `build_index.py`

A fresh run differs from the committed `jd_embeddings.npy` (max abs diff 0.095; cause undetermined; the re-embedding itself is deterministic). Reproduction must route the rebuilt copy elsewhere (`--out-jd-embeds`), and a fully fresh pipeline may rank slightly differently.

## C. Honeypot rules

- **Breadth.** 5,542 of 100K flagged (5.5%) vs ≈ 80 expected *(project notes)*; rule 6 (education order) accounts for 5,248, rule 2 for 295, rule 1 for 21, and rules 4 and 5 never fire. 177 of the 2,969 shortlisted candidates are removed. Rule 6 deserves a manual sample review for false positives (e.g. dual/part-time degrees, differently named degrees).
- **No company-timeline check** any more (by design — see rule 3 history).
- **Rule 4** compares only adjacent completed roles after sorting and ignores current roles.
- **Real recall unknown:** zero flagged in the top 100 says little about how many of the ≈ 80 true honeypots the filter catches.

## D. Reasoning

- Default text is template-based: 5 × 4 frames, so phrasing repeats across rows; claims are verified, wording is not rich.
- LLM rewrites cover ≈ 38–39% of rows; models often drop the verbatim quote even when told to, so those rewrites are rejected.
- **Cache fragility:** entries are keyed by the *exact draft text*. Any change to `compose_reasoning`, to a candidate's rank or to the ranking itself silently turns hits into misses (falling back to the grounded text). Rebuild the cache after changing the ranker or the draft generator.
- **Slotted mode is unfinished:** `reasoning_slotted.py`'s docstring mentions an offline `slotted-llm` build mode, but `rank.py` only exposes `slotted-cached`; `generate_reasonings_slotted` must be invoked from Python, and the committed cache holds only rewrite entries (no templates), so `slotted-cached` always falls back.
- Model weights are downloaded from the Hub on first use and are not revision-pinned.
- The legacy path (`--reasoning-mode llm`) remains but is generic and unverified.

## E. App / ops

- Cosmetic progress bar; **hard-coded KPI values** (2,956 / 2,556) are stale — current figures are 2,969 shortlisted / 2,792 post-honeypot *(project notes)*.
- `@st.cache_data` on `load_data(csv_path)` is keyed by path; re-running the pipeline overwrites `temp_preview_ranked.csv` but the path string is unchanged, so the Explorer can show stale rows until the cache is cleared.
- `temp_preview_ranked.csv` is written to the CWD and is shared across all sessions of a deployed Space.
- Sandbox mode works for ≤ 1,500 IDs from the pool; larger uploads need the git-ignored indexes.
- CI force-pushes to the Space on every `main` push and runs no tests.
- Pickled artifacts (`bm25.pkl`, `ranker_model.pkl`) execute code on load.
- Offline runs require MiniLM to be in the local HF cache beforehand.
- With few CPU cores the ranking step may exceed 5 minutes (≈ 430 CPU-seconds of work; ≥ ~3.6 min even with perfect scaling on 2 cores).

## F. Documentation drift (remaining)

The top-level README and architecture notes were rewritten and are now largely accurate. What still differs or is stale:

| Source | Says | Reality |
|---|---|---|
| `redrob_ranking_architecture.md` §4 layout | tests listed: `test_reasoning_grounded.py`, no-stuffing, streamlit | there are also `test_reasoning_llm_rewrite.py` and `test_reasoning_slotted.py`; `reasoning_llm_rewrite.py`, `reasoning_slotted.py`, `artifacts/reasoning_rewrites.json` are not in the layout |
| `redrob_ranking_architecture.md` §14 | "34 commits … the 2026-09-20 changes, which are not yet committed" | changes are now committed on `main` |
| `README.md` pipeline box | lists 5 stages of offline scripts, no mention of rewrite cache | optional `grounded-llm` build exists |
| `reasoning_slotted.py` docstring | offline build "slotted-llm" | no such `--reasoning-mode` choice |
| `schema_utils.py`, `honeypot_flags.py` docstrings | reference `ARCHITECTURE.md` / "spec's example" | file is `redrob_ranking_architecture.md` |
| `validate_local.py` tone check | warns rank > 70 rows lacking "gap/concern/limited…" | grounded text says "Caveat:"/"Held back by" ⇒ false positives |
| `app/app.py` KPI cards | 2,956 / 2,556 | 2,969 / 2,792 |

## G. Roadmap / good first issues

**Quick wins**

1. Fix A1 (carry original row position) and add a subset-vs-full regression test.
2. Make `validate_local.validate` always return `(errors, warnings)`; fix its tone check to recognise "Caveat"/"Held back by".
3. Replace substring checks with word-boundary matching; add unit tests for each `compute_*` rule in `features_utils.py` and for `compute_availability_multiplier` / `apply_availability_multiplier` (including the negative-score branch).
4. Add a `StandardScaler` to the Ridge pipeline and report coefficients on standardised features.
5. Replace hard-coded dashboard KPIs with values emitted by `rank.py` (e.g. a JSON sidecar).
6. Add a CI job that runs `pytest` before the HF sync; make the sync depend on it.
7. Implement the title-mismatch top-10 test hinted at by unused constants in `tests/test_no_stuffing_regression.py`.

**Medium**

8. Regenerate `eval_set.csv` and retrain after the rule-3 removal; compare coefficients and top-100 overlap before/after.
9. Enforce the three ≈ 0-weight JD disqualifiers explicitly (a post-model multiplier like availability, with a documented parameter).
10. Per-rule honeypot review (esp. rule 6) and tune thresholds from distributions.
11. Pairwise "obviously better/worse" regression set for the ranker, and per-sample-type CV metrics.
12. Expose behavioural components to the ranker; fit their weights.
13. Multi-query retrieval (one query per JD requirement) with reciprocal-rank fusion for recall.
14. Vectorise `iterrows()` hot spots (`extract_candidate_features`, `honeypot_flags`, availability `.apply`).
15. Finish the slotted-template build path (CLI mode + populated cache) or remove the module.

**Larger**

16. Human-reviewed relevance labels; measure agreement between rules and reviewers.
17. Cross-encoder or LLM re-ranking of the top ~50 under the time budget.
18. Investigate why a fresh JD embedding differs from the committed one and make `build_index.py` reproduce it.
19. Cache the Qwen/MiniLM weights in a build image for truly offline runs.
