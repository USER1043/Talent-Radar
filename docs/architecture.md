# Architecture

## The core idea: spend time offline, be careful online

The grader runs `rank.py` under hard limits (≤ 5 min, ≤ 16 GB, CPU-only, no network). Everything that does not depend on the JD *at ranking time* is
precomputed once and cached under `artifacts/`.

```
OFFLINE (unlimited time, scripted, reproducible)         ONLINE — rank.py (graded)
────────────────────────────────────────────────         ──────────────────────────────────────────────
parse_candidates.py                                      1  Scan candidates file for IDs (regex)
  JSONL(.gz) ──► features.parquet  (flat, 48 cols)       2  Load parquet artifacts, filter to IDs
honeypot_flags.py                                        3  Hybrid retrieval: FAISS ∪ BM25  (≤ 3,000)
  5 OR-logic rules ──► honeypot_flags.parquet               (skipped when ≤ 1,500 candidates)
build_index.py                                           4  Structured JD-rule + behavioural features
  MiniLM ──► faiss.index, bm25.pkl, jd_embeddings.npy    5  Embed shortlist, cosine to both JD vectors
label_eval_set.py ──► eval_set.csv (150 rows)            6  Ridge score × availability multiplier
train_ranker.py   ──► ranker_model.pkl                   7  Drop honeypots (hard filter)
reasoning_llm_rewrite.py (optional, offline)             8  Sort, tie-break, take top 100
  ──► reasoning_rewrites.json                            9  Reasoning (grounded by default)
(company_stats.py: retired, kept for reference)          10 Write CSV
```

## Why this shape?

The scoring formula is `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. 80% of the score is the quality of the **top 10 and top 50**. That pushes the
design toward *cheap broad recall, then careful scoring of a small set*: embedding 100K narratives is paid offline; online only the shortlist (≈ 3K) is embedded and scored by a tiny linear model.

Measured (16 cores, offline, default mode): **1 min 47 s wall, 4.0 GB peak**; roughly 390% average CPU (~430 CPU-seconds). With few cores the
run is slower — even with perfect scaling a 2-core machine needs ≥ ~3.6 min. Rough split: stages 1–4 ≈ 1 min, stage 5 ≈ 50 s, stage 9 ≈ 0.1 s.

## Data flow in `rank.py`

```
candidates.jsonl ──regex──► active_ids (set of CAND_xxxxxxx)
                                  │
features.parquet ──filter(active_ids)──► df_feat
honeypot_flags.parquet ─filter──────────► df_hp
                                  │
jd_embeddings.npy  [0]=literal JD, [1]="ideal candidate" paragraph
                                  │
                       ≤ 1,500 active IDs?
                  yes ┌────────────┴────────────┐ no
   score every candidate;            FAISS (IndexFlatIP, 384-d, normalised ⇒ cosine)
   bm25_score := 0                     query = ideal JD, top 1500 ─┐
                                     BM25Okapi, query = literal JD tokens, top 1500 ─┤ union ≤ 3,000
                                                                                     ▼
                                                                          shortlist_df
   extract_candidate_features(row) ×N ──► 13 structured features                     │
   MiniLM.encode(narrative_text) ──► cos(JD), cos(ideal)   (+ bm25_score)            │
                                                                                     ▼
   X = 16 features, notice_period_penalty forced to 0 ─► Ridge.predict ─► model_score
   availability_multiplier(row) ─► score = model_score × m  (÷ m if negative)
                                                                                     ▼
   merge honeypot flags ─► drop is_honeypot ─► sort(score desc, id asc) ─► head(100)
                                                                                     ▼
   reasoning (mode) ─► submission.csv  (candidate_id, rank, score, reasoning)
```

## Stage reference (`rank.py`)

| # | Log label | What happens | Notes |
|---|---|---|---|
| 1 | "Loading candidate IDs…" | Regex `"candidate_id"\s*:\s*"(CAND_\d{7})"` per line; **no JSON parsing** | Streams the 465 MB file cheaply; `.gz` supported |
| 2 | "Loading cached artifacts…" | Reads features + honeypot parquet; filters to `active_ids` and resets index | IDs absent from `features.parquet` are silently ignored; none matching ⇒ empty CSV |
| 3 | "Performing hybrid retrieval…" | FAISS top-1500 (ideal-candidate embedding) ∪ BM25 top-1500 (literal JD tokens). **Small-sample mode** (≤ 1,500 IDs): skip retrieval, score everyone, `bm25_score = 0`, `faiss.index`/`bm25.pkl` not needed | Shortlist ≈ 2,969 on the full pool |
| 4 | "Computing structured candidate features…" | `extract_candidate_features` per shortlist row | See [retrieval-and-ranking](retrieval-and-ranking.md) |
| 5 | "Embedding shortlist…" | `all-MiniLM-L6-v2` on CPU encodes `narrative_text`; dot product with both JD vectors; BM25 score looked up | Dominant CPU cost; MiniLM must already be in the local HF cache for offline runs |
| 6 | "Scoring candidates…" | Loads `{"type","model","feature_names"}`; sets `notice_period_penalty` to 0 in `X`; `model_score = predict(X)`; `score = apply_availability_multiplier(model_score, m)` | Prints mean multiplier and how many fall below 0.8 |
| 7 | "Filtering out honeypots…" | Inner-merge on `candidate_id`, keep `is_honeypot == False` | After scoring, shortlist only |
| 8 | "Sorting, selecting top 100…" | Sort by `score` desc, numeric part of ID asc | Mirrors the validator's tie-break rule |
| 9 | "Generating reasoning…" | Dispatch on `--reasoning-mode` | Default `grounded`: deterministic, audited, ~0.1 s |
| 10 | "Wrote final ranking list…" | CSV `candidate_id, rank, score, reasoning`; prints total time | `rank.py` does not run the validators itself |

### CLI

```
python rank.py --candidates FILE --out CSV
    [--features P] [--honeypots P] [--faiss-index P] [--bm25 P] [--jd-embeds P] [--job-description P] [--ranker-model P]
    [--reasoning-mode {grounded,grounded-cached,grounded-llm,slotted-cached,llm}]   # default grounded
    [--rewrite-cache artifacts/reasoning_rewrites.json] [--llm-model Qwen/Qwen2.5-0.5B-Instruct] [--reasoning-diagnostics FILE.json]
```

Reasoning-mode details: [reasoning-engine.md](reasoning-engine.md). Measured performance by mode (full pool, 16 cores):

| Mode | Wall time | Peak RAM |
|---|---|---|
| `grounded` (default) | 1 min 47 s | 4.0 GB |
| `grounded-cached` | 1 min 54 s | 4.0 GB |
| `llm` (legacy Qwen 0.5B) | 3 min 47 s | 7.0 GB |
| `grounded-llm` (offline cache build) | 1 min 47 s + ~6 min (0.5B) / ~17 min (1.5B) for 100 rows | 5.7 GB / ≈ 7 GB |

## Module dependency graph

```
rank.py ──► build_index.py        (tokenize, extract_jd_texts)
        ├─► features_utils.py     (extract_candidate_features, availability multiplier, AS_OF_DATE)
        ├─► reasoning_grounded.py (default; also imports AS_OF_DATE from features_utils)
        ├─► reasoning_llm_rewrite.py (lazy; grounded-llm / grounded-cached)
        │       └─► reasoning_grounded.py
        ├─► reasoning_slotted.py  (lazy; slotted-cached) ─► reasoning_grounded.py, reasoning_llm_rewrite.py
        └─► reasoning.py          (lazy; legacy llm mode — imports torch)

train_ranker.py ──► build_index.py, features_utils.py
parse_candidates.py, honeypot_flags.py, company_stats.py ──► schema_utils.py
app/app.py ──► features_utils.py (display), subprocess → rank.py
```

`rank.py` imports from `build_index.py`, so importing `rank` also imports `sentence_transformers`/`faiss` at module level — it is a script, not a library.
`torch`/`transformers` are only imported on the LLM paths.

## Performance and scaling notes

- Retrieval is **exact** (flat inner-product index), not approximate — at 100K × 384 floats that is fast and removes ANN recall as a variable.
- `iterrows()` is used in several places (feature extraction, honeypot flags, availability multiplier via `.apply`). Acceptable at ≈ 3K rows online and 100K offline; vectorise before growing the shortlist.
- Precompute: `build_index.py` ≈ 37 min (embedding 100K profiles on 16 cores), `parse_candidates.py` ≈ 20 s, `honeypot_flags.py` ≈ 5 s.
