---
title: Talent Radar — Candidate Explorer
emoji: 🎯
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.13"
app_file: app/app.py
pinned: false
short_description: Visual explorer for AI/ML candidate rankings.
---

# Talent Radar — Redrob Hackathon Submission (team AI_WRAPPERS)

Ranks the top 100 of a 100,000-candidate pool for the Redrob *Senior AI Engineer (Founding Team)*
job description and writes `submission.csv` with a fact-checked reasoning for every finalist.
The ranking step is CPU-only, makes no network or LLM-API calls, and runs in under two minutes.

---

## Reproduce the submission

```bash
# 0. Get the code and the two Git-LFS data files
git clone https://github.com/USER1043/talent-radar && cd talent-radar
git lfs install && git lfs pull            # artifacts/features.parquet, artifacts/honeypot_flags.parquet

# 1. Python 3.13 environment (CPU only, no GPU needed)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # tested route: `uv pip install -r requirements.txt --torch-backend cpu`

# 2. Data: put the 100K pool next to rank.py (git-ignored; .jsonl or .jsonl.gz both work)
cp /path/to/candidates.jsonl ./candidates.jsonl

# 3. Precomputed search indexes (NOT committed: 153 MB + 186 MB). One-off, about 37 minutes on 16 CPU cores.
python build_index.py --out-jd-embeds artifacts/jd_embeddings.rebuilt.npy     # see "JD embeddings" note below

# 4. The graded ranking step (about 1 min 47 s wall, 4 GB peak on 16 cores)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
#    same, using the precomputed LLM rewrites (no model needed; 1 min 54 s):
#    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --reasoning-mode grounded-cached

# 5. Validate the format
python validate_submission.py submission.csv
```

**No network at ranking time.** `rank.py` loads `sentence-transformers/all-MiniLM-L6-v2` by name, so it must be in the local
Hugging Face cache before an offline run (`python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"`,
then run with `HF_HUB_OFFLINE=1`). The default reasoning mode needs no other model.

**JD embeddings.** `rank.py` reads the committed `artifacts/jd_embeddings.npy`, which the trained ranker was fit against.
`build_index.py` also writes that file by default, but re-embedding the job description gives *different* values
(max absolute difference 0.095; cause not determined; the re-embedding itself is deterministic), which shifts scores slightly.
That is why step 3 sends the rebuilt copy to a different path.

### Regenerating the other artifacts (optional)

| Step | Command | Time | Notes |
|---|---|---|---|
| Features | `python parse_candidates.py --candidates ./candidates.jsonl --out ./artifacts/features.parquet` | ~20 s | Verified to reproduce the committed file exactly |
| Honeypot flags | `python honeypot_flags.py --features ./artifacts/features.parquet --out ./artifacts/honeypot_flags.parquet` | ~5 s | 5,542 candidates flagged (5.5%) |
| Indexes | `python build_index.py …` (step 3 above) | ~37 min | Writes `faiss.index`, `bm25.pkl` |
| Eval labels | `python label_eval_set.py` | not timed | **Overwrites `artifacts/eval_set.csv`** with rule-generated labels; needs `faiss.index` and the honeypot flags |
| Ranker | `python train_ranker.py` | not timed | **Overwrites `artifacts/ranker_model.pkl`**; 5-fold CV of Ridge vs LightGBM on the eval set, keeps the better |

`company_stats.py` is no longer part of the pipeline (see "Honeypot filter").

---

## Pipeline

```
OFFLINE (scripted, no time limit)                ONLINE: rank.py (graded, <= 5 min)
─────────────────────────────────                ───────────────────────────────────────────
parse_candidates.py                              1  read candidate IDs
  JSONL -> features.parquet (48 cols)            2  load features, honeypot flags, indexes
honeypot_flags.py                                3  hybrid retrieval: FAISS top-1500 (ideal-candidate
  5 rules -> honeypot_flags.parquet                 embedding) U BM25 top-1500 (JD text) = shortlist
build_index.py                                      (<= 3,000; 2,969 on the full pool)
  MiniLM embeddings -> FAISS + BM25              4  structured JD-rule + behavioural features
label_eval_set.py -> eval_set.csv (150 rows)     5  embed shortlist, similarity to both JD texts
train_ranker.py -> ranker_model.pkl              6  Ridge score, then availability multiplier
                                                 7  drop honeypot-flagged candidates
                                                 8  sort, top 100, ties by candidate_id ascending
                                                 9  reasoning (fact-grounded, audited)
                                                 10 write CSV
```

### What each stage does

- **Hybrid retrieval.** Dense (FAISS `IndexFlatIP` over `all-MiniLM-L6-v2` embeddings of each candidate's narrative, queried with the
  JD's "ideal candidate" paragraph) plus lexical (BM25Okapi, queried with the literal JD text). The union of two top-1500 lists is at
  most 3,000 candidates.
- **Ranker.** A Ridge regression over 16 features: semantic similarity to the literal JD and to the ideal-candidate paragraph, BM25 score,
  years of experience, skill count, skill-to-experience ratio, a composite `behavioral_score` (recency, response rate and speed, interview
  completion, offer acceptance, open-to-work, verification, GitHub), and JD-rule features (`no_recent_coding`, `no_external_validation`,
  `cv_speech_robotics_no_nlp`, `framework_enthusiast_signal`, `title_chasing_pattern`, `is_consulting_only`, `is_pure_research`,
  `is_recent_wrapper_only`, `notice_period_penalty`).
- **Notice period.** The fitted Ridge gave `notice_period_penalty` a *positive* weight (+0.16), i.e. it rewarded long notice periods. That
  term is neutralised at prediction time and notice is instead penalised in the availability multiplier below.
- **Availability multiplier** (applied to the Ridge score after prediction; range 0.29 to 1.0), from the JD's "a perfect-on-paper candidate
  who hasn't logged in for 6 months and has a 5% recruiter response rate is not actually available":
  recency (no penalty up to 90 days idle, falling to x0.7 at 180 days), recruiter response rate (no penalty at 40% or more, x0.5 at 0%),
  and notice period (x1.0 / 0.96 / 0.90 / 0.84 for <=30 / <=60 / <=90 / >90 days). The parameters are judgment values, not fitted.
- **Honeypot filter.** OR-logic rules that use only fields inside a candidate's own record; any hit removes the candidate before top-100
  selection: (1) an advanced/expert skill with 0 months of use, (2) at most 1 year of experience with 8 or more skills, (4) overlapping
  roles, (5) `duration_months` inconsistent with the dates, (6) an impossible education order. A former rule 3 ("tenure predates the
  company's inferred founding") was **removed**: the schema has no founding date, so it was inferred from other candidates' start dates in
  the pool and flagged about 17% of all candidates regardless of role.
- **Reasoning.** `reasoning_grounded.py` builds each two-sentence explanation *only* from that candidate's own record (default mode). It
  is a deterministic generator that fills sentence frames with verified facts: title, employer, years; up to three JD-relevant skills the
  candidate actually lists, with proficiency and months; a verbatim quote from a role description; and the candidate's real concerns
  (low response rate, inactivity, notice period, location, experience outside the JD's 5-9 year band, missing retrieval skills, and so on).
  Wording follows the rank tier. `audit_reasoning()` re-checks every claim against the record and falls back to a minimal true sentence on
  any failure. Because it is template-based, some phrasing repeats across rows (for example "the JD's 5-9 year band"). The previous
  Qwen2.5-0.5B generator (`reasoning.py`, abstract labels only) is still available with `--reasoning-mode llm`.
  **LLM polish, run offline:** `--reasoning-mode grounded-llm` lets a local model reword each grounded draft under a per-row KEEP checklist, a forced correct
  "Title at Company" opening and worked examples; a strict gate accepts a rewrite only if it keeps the title/company relation, at least two skills exactly, every verbatim quote,
  every concern and availability number and the rank-tier wording, adds no number, name, evaluation word or job-posting phrasing, and is grammatical. Accepted rewrites are written to
  `artifacts/reasoning_rewrites.json`, keyed by the exact draft. `--reasoning-mode grounded-cached` only *reads* that cache and re-checks every entry with the same gate at ranking time, so
  **no model is loaded in the graded step**; a cache miss or failed check falls back to the grounded text. Result (`submission_v5.csv`): 38 of the 100 finalists are reworded (31 by Qwen2.5-1.5B-Instruct,
  7 by Qwen2.5-0.5B-Instruct), all 42 verbatim quotes and 3.0 skills per row kept, 0 grounding failures over 719 claims checked against the raw data, boilerplate 16.2% -> 7.7%. Both models drop the
  verbatim quote in most rows even when told explicitly (the 1.5B model kept it in only 3 of 42), which caps coverage; only rewrites that keep it are used. Build the cache with e.g.
  `LLM_DTYPE=bfloat16 python rank.py ... --reasoning-mode grounded-llm --llm-model Qwen/Qwen2.5-1.5B-Instruct --rewrite-cache artifacts/reasoning_rewrites.json` (about 17 minutes for 100 rows, offline;
  bfloat16 halves the model's memory). Details: `snapshots/2026-09-20_v5_llm_rewrite_prompted/RESULTS.md` (kept outside the repo).

---

## Artifacts

| File | In repo | Produced by |
|---|---|---|
| `artifacts/features.parquet` (87.5 MB) | yes (Git LFS) | `parse_candidates.py` |
| `artifacts/honeypot_flags.parquet` | yes (Git LFS) | `honeypot_flags.py` |
| `artifacts/jd_embeddings.npy` | yes | `build_index.py` (see the JD-embeddings note) |
| `artifacts/eval_set.csv` (150 rows) | yes | `label_eval_set.py` |
| `artifacts/ranker_model.pkl` (Ridge) | yes | `train_ranker.py` |
| `artifacts/reasoning_rewrites.json` (LLM rewrites, keyed by draft) | yes | `rank.py --reasoning-mode grounded-llm` (offline; optional) |
| `artifacts/faiss.index` (153.6 MB), `artifacts/bm25.pkl` (186 MB) | **no** (git-ignored) | `build_index.py` |
| `candidates.jsonl` | **no** (git-ignored) | provided by the organisers |

---

## Measured performance

Full 100,000-candidate run, offline, 16 CPU cores, about 14 GiB RAM, Python 3.13:

| Mode | Wall time | Peak memory |
|---|---|---|
| default (`--reasoning-mode grounded`) | **1 min 47 s** | **4.0 GB** |
| `--reasoning-mode llm` (Qwen2.5-0.5B) | 3 min 47 s | 7.0 GB |
| `--reasoning-mode grounded-cached` (reads the precomputed rewrites, no model) | **1 min 54 s** | **4.0 GB** |
| `--reasoning-mode grounded-llm` (offline cache build, live model) | 1 min 47 s + 6 min (0.5B) or + 17 min (1.5B) for 100 rows | 5.7 GB (0.5B), about 7 GB (1.5B, bfloat16) |

Average CPU use is about 390% (roughly 430 CPU-seconds), so a grader with fewer cores will be slower: even with perfect scaling a 2-core
machine needs at least about 3.6 minutes, and real scaling is worse. Precompute: `build_index.py` about 37 min, `parse_candidates.py` 20 s, `honeypot_flags.py` 5 s.

---

## Known limitations (read before trusting the ranking)

- **No ground truth.** `artifacts/eval_set.csv` labels are **generated by rules in `label_eval_set.py`, not hand-labeled**: 30 flagged
  honeypots -> 0, 20 random background candidates -> 0 or 1, 40 "keyword stuffers" (non-ML title, high similarity) -> 1, 20 mild fits -> 3,
  40 "strong fits" (ML title, high similarity, not consulting-only) -> 5. The Ridge therefore learns to reproduce those heuristics, and its
  cross-validation score measures agreement with them, not with the organisers' hidden relevance tiers.
- **Some JD disqualifiers carry almost no weight.** In the fitted Ridge, `is_pure_research`, `is_recent_wrapper_only` and `is_consulting_only`
  have coefficients of about 0, so the model does not really penalise them (`no_recent_coding` -0.76 and `no_external_validation` -0.61
  do have effect). Several coefficients have unintuitive signs (for example `semantic_sim_to_jd` -0.87).
- **Honeypot filter is broad and partly inert.** On this pool rules 4 and 5 never fire, rule 6 flags 5,248 candidates (5.2%), and about
  80 real honeypots are expected. Rule 1 (21 flags) and rule 2 (295) are the ones that resemble the spec's examples.
- **Tightly packed scores.** In the top 100 the scores from rank 50 to rank 100 span only about 0.4 points, so small modifier changes move
  mid-list candidates a long way. The availability multiplier's parameters are judgment values.
- **JD embeddings** differ between the committed file and a fresh `build_index.py` run (see above).
- **Shortlist size** is at most 3,000 (2,969 in practice), not the 4-5K quoted in earlier drafts.
- **Only pool candidates can be ranked.** `rank.py` uses the precomputed `artifacts/features.parquet`; IDs in the candidates file that are not in it
  are silently ignored, and if none match it writes an empty CSV. A sandbox sample must therefore be drawn from the 100K pool.
- **Small-sample (sandbox) mode.** With 1,500 or fewer candidates `rank.py` skips FAISS/BM25 retrieval and scores every candidate, so it runs without
  `faiss.index` / `bm25.pkl` (which are not in the repo or on the Hugging Face Space). The BM25 feature is set to 0 in that mode (ranker weight ~0.004),
  so scores can differ slightly from a full-pool run. A 20-candidate file ranks in about 5 seconds.

---

## Tests, validation and sandbox

```bash
pip install pytest && pytest tests          # 29 pass, 1 setup error (needs sample_submission.csv, see below)
python validate_submission.py submission.csv   # the organisers' format validator
python validate_local.py --csv submission.csv --candidates ./candidates.jsonl   # extra local checks
streamlit run app/app.py                    # sandbox: Pipeline Runner + Candidate Explorer
```

`tests/test_no_stuffing_regression.py` expects the organisers' `sample_submission.csv` in the repo root (it is not committed), so it
errors in a fresh clone. `validate_local.py` ends with a crude tone check that warns when a rank>70 row lacks words such as "gap", "concern"
or "limited"; the grounded reasoning states its concerns with other wording ("Caveat:", "Held back by"), so that warning can be a false
positive. The Hugging Face Space (`HuggingFace1043/talent-radar-redrob-sandbox`) is updated by
`.github/workflows/sync_to_hf.yml` on every push to `main`.
