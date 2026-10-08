# Architecture

## The core idea: spend time offline, be careful online

The grader runs `rank.py` under hard limits (≤ 5 min, ≤ 16 GB, CPU-only, no
network). Everything that does not depend on the JD *at ranking time* is
therefore precomputed once and cached under `artifacts/`.

```
OFFLINE (unlimited time, scripted, reproducible)        ONLINE — rank.py (graded)
────────────────────────────────────────────────        ──────────────────────────────────────
parse_candidates.py                                      1  Scan candidates file for IDs (regex)
  JSONL(.gz) ──► features.parquet  (flat, 1 row/cand)    2  Load parquet artifacts, filter to IDs
company_stats.py                                         3  Hybrid retrieval: FAISS ∪ BM25
  career_history ──► company_stats.parquet               4  Structured JD-rule + behavioural feats
honeypot_flags.py                                        5  Embed shortlist, cosine to JD vectors
  6 OR-logic rules ──► honeypot_flags.parquet            6  Score with ranker_model.pkl (Ridge)
build_index.py                                           7  Drop honeypots (hard filter)
  MiniLM ──► faiss.index, bm25.pkl, jd_embeddings.npy    8  Sort, tie-break, take top 100
label_eval_set.py ──► eval_set.csv (150 rows)            9  Generate reasoning (plan → Qwen/fallback)
train_ranker.py   ──► ranker_model.pkl                   10 Write CSV
```

## Why this shape?

The scoring formula is `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.
80% of the score is the quality of the **top 10 and top 50**. That pushes the
design toward *cheap broad recall, then expensive careful scoring of a small
set*: embedding 100K narratives is paid offline; online we only embed the
~3K shortlist and run a tiny linear model.

## Data flow in `rank.py`

```
candidates.jsonl ──regex──► active_ids (set of CAND_xxxxxxx)
                                  │
features.parquet ──filter(active_ids)──► df_feat (row order = FAISS/BM25 order*)
honeypot_flags.parquet ─filter──────────► df_hp
                                  │
jd_embeddings.npy  [0]=literal JD, [1]="ideal candidate" paragraph
                                  │
FAISS (IndexFlatIP, 384-d, L2-normalised ⇒ cosine) ── query = ideal JD ── top 1500 ─┐
BM25Okapi (pickled)               ── query = literal JD tokens        ── top 1500 ─┤ union ≈ 3K
                                                                                   ▼
                                                                        shortlist_df
   extract_candidate_features(row) ×N  ──► 13 structured features                   │
   MiniLM.encode(narrative_text)   ──► cos(JD), cos(ideal)  (+ bm25_score)           │
                                                                                      ▼
   X = [16 features] ──► Ridge.predict ──► score                           (* see known issues)
                                                                                      ▼
   merge honeypot flags ─► drop is_honeypot ─► sort(score desc, id asc) ─► head(100)
                                                                                      ▼
   generate_reasonings(top_100)  ──► submission.csv
```

## Stage reference (`rank.py`)

| # | Log label | What happens | Notes |
|---|---|---|---|
| 1 | "Loading candidate IDs…" | Regex `"candidate_id"\s*:\s*"(CAND_\d{7})"` over each line; **no JSON parsing** | Streams the 465 MB file cheaply; supports `.gz` via suffix |
| 2 | "Loading cached artifacts…" | Reads features + honeypot parquet; filters both to `active_ids` and resets index | Reset index is what FAISS/BM25 row positions are later compared with |
| 3 | "Performing hybrid retrieval…" | FAISS top-1500 using **ideal-candidate** embedding; BM25 top-1500 using **literal JD** tokens; union | If ≤ 1500 active IDs, retrieval is skipped and all are scored |
| 4 | "Computing structured candidate features…" | `extract_candidate_features` per shortlist row (`iterrows`) | See [retrieval-and-ranking](retrieval-and-ranking.md) |
| 5 | "Embedding shortlist…" | `all-MiniLM-L6-v2` on CPU encodes `narrative_text`; dot product with both JD vectors; BM25 score looked up from the full-pool score array | Dominant CPU cost after Qwen |
| 6 | "Scoring candidates…" | Loads `{"type","model","feature_names"}` pickle; `predict(X)` | Feature order is dictated by the pickle, not by code |
| 7 | "Filtering out honeypots…" | Inner-merge on `candidate_id`, keep `is_honeypot == False` | Applied *after* scoring, only to shortlist |
| 8 | "Sorting, selecting top 100…" | Sort by `score` desc, numeric part of ID asc | Mirrors validator tie-break rule |
| 9 | "Generating reasoning…" | `reasoning.generate_reasonings(top_100)` | Batched Qwen with deterministic fallback |
| 10 | "Wrote final ranking list…" | CSV with columns `candidate_id, rank, score, reasoning` | Prints total runtime |

Edge cases handled explicitly: empty shortlist → writes an empty CSV with the right
header and returns; `.gz` input; score ties.

## Module dependency graph

```
rank.py ──► build_index.py (tokenize, extract_jd_texts)
        ├─► features_utils.py (extract_candidate_features)
        └─► reasoning.py      (generate_reasonings)

train_ranker.py ──► build_index.py, features_utils.py
parse_candidates.py ──► schema_utils.py
company_stats.py     ──► schema_utils.py
honeypot_flags.py    ──► schema_utils.py
app/app.py ──► features_utils.py (display), subprocess → rank.py
```

`rank.py` imports from `build_index.py`, so importing `rank` also imports
`sentence_transformers`/`faiss` at module level — it is intentionally a script,
not a library.

## Concurrency and performance notes

- Retrieval is **exact** (flat inner-product index), not approximate — at 100K × 384 floats that is fast enough and removes ANN recall as a variable.
- `iterrows()` is used in several places. It is acceptable at ~3K rows online and 100K rows offline; vectorise before growing the shortlist.
- `torch.set_num_threads(8)` is hard-coded in `reasoning.py`.
- All 100 Qwen prompts are generated in **one batch** (`max_new_tokens=85`, greedy decoding, left padding).
