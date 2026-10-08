# Redrob Hackathon — Ranking System Architecture (v2.0, as built)

This document describes the system **as it is implemented and measured** (last verified 2026-09-20). It replaces the v1.0 plan,
which was written before the code existed; section 16 lists what changed from that plan. For exact reproduction commands see
`README.md`.

---

## 0. Grounded in

- `job_description.docx` — JD text, explicit disqualifiers, "ideal candidate" narrative, anti-keyword-stuffing warning.
- `submission_spec.docx` — CSV schema, compute limits, scoring formula, Stage 1-5 pipeline, reasoning-audit checks.
- `candidate_schema.json` — candidate record structure (note: it has **no company-founding field**).
- `redrob_signals_doc.docx` — the 23 behavioural signal fields (the doc contains only the field table, despite the bundle README saying it
  also explains traps).
- `sample_submission.csv` — a worked example of the keyword-stuffing trap.
- Bundle `README.docx` — trap categories: keyword stuffers, plain-language Tier 5s, behavioural twins, ~80 honeypots.

---

## 1. Mission and what the scoring formula prioritises

```
composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

80% of the score depends on the top 10 and top 50 being in the right order, so precision at the top matters more than the tail. The
spec also states that honeypots need no special-casing ("a good ranking system naturally avoids them"); the only hard rule is that more
than 10% honeypots in the top 100 disqualifies a submission.

---

## 2. Hard constraints (ranking step only) and measured results

| Constraint | Limit | Measured (default mode, 16 cores, offline) |
|---|---|---|
| Runtime | <= 5 min wall clock | **1 min 47 s** (legacy `--reasoning-mode llm`: 3 min 47 s) |
| Memory | <= 16 GB | **4.0 GB peak** (legacy llm: 7.0 GB) |
| Compute | CPU only | yes (CPU-only torch build) |
| Network | none, no hosted LLM | none; MiniLM must already be in the local cache |
| Honeypots in top 100 | <= 10% | 0 flagged by the pipeline's own rules |

Average CPU use is about 390% (about 430 CPU-seconds). On a grader with fewer cores the run is slower: even with perfect scaling, 2 cores
need at least about 3.6 minutes. Precompute (unlimited time): `build_index.py` about 37 min, `parse_candidates.py` 20 s, `honeypot_flags.py` 5 s.

---

## 3. Pipeline as built

```
OFFLINE (scripted)                              ONLINE: rank.py (graded)
──────────────────                              ─────────────────────────────────────────────
parse_candidates.py                             1  read candidate IDs from the candidates file
  JSONL -> artifacts/features.parquet           2  load features, honeypot flags, FAISS, BM25, ranker
honeypot_flags.py                               3  hybrid retrieval -> shortlist (<= 3,000)
  5 rules -> honeypot_flags.parquet             4  structured JD-rule + behavioural features
build_index.py                                  5  embed shortlist; similarity to literal JD and ideal paragraph
  MiniLM -> faiss.index, bm25.pkl,              6  Ridge score (notice term neutralised) x availability multiplier
  jd_embeddings.npy                             7  drop honeypot-flagged candidates
label_eval_set.py -> eval_set.csv               8  sort, take top 100, ties by candidate_id ascending
train_ranker.py -> ranker_model.pkl             9  reasoning: fact-grounded generator (default) or legacy Qwen
                                                10 write CSV (candidate_id, rank, score, reasoning)
```

Approximate time split in the default run (from stage timestamps): stages 1-4 about 1 min, stage 5 about 50 s, stage 9 about 0.1 s (measured).

---

## 4. Repository layout (flat)

```
README.md, requirements.txt, submission_metadata.yaml, job_description.docx
rank.py                     the graded entry point
parse_candidates.py, honeypot_flags.py, build_index.py, label_eval_set.py, train_ranker.py   offline steps
features_utils.py           JD-rule + behavioural features, availability multiplier, AS_OF_DATE
schema_utils.py             date helpers, CONSULTING_FIRMS
reasoning_grounded.py       default reasoning generator + audit
reasoning.py                legacy Qwen/reason-planner generator (--reasoning-mode llm)
company_stats.py            no longer used by the pipeline (was the input to the removed honeypot rule 3)
validate_local.py, validate_submission.py    local checks, and the organisers' format validator
app/app.py                  Streamlit sandbox (Pipeline Runner + Candidate Explorer)
tests/                      test_reasoning_grounded.py, test_no_stuffing_regression.py, test_streamlit_app.py
artifacts/                  features.parquet*, honeypot_flags.parquet*, jd_embeddings.npy, eval_set.csv, ranker_model.pkl
                            (* Git LFS). faiss.index and bm25.pkl are generated and git-ignored.
.github/workflows/sync_to_hf.yml    pushes to the Hugging Face Space on every push to main
```

---

## 5. Candidate feature extraction (`parse_candidates.py`)

One flattened row per candidate (48 columns), streamed from the JSONL; regenerating the file reproduces the committed one exactly.
Notable columns: `years_of_experience`, `current_title`, `current_company`, `location`, `country`, `n_employers`, `avg_tenure_months`,
`tenure_consistency_mismatches`, `is_consulting_only` (every employer is one of TCS / Tata Consultancy Services / Infosys / Wipro /
Accenture / Cognizant / Capgemini, the JD's named list), `has_product_company_experience` (any employer outside that list), `n_skills`,
`n_expert_or_advanced_skills`, `n_zero_duration_high_proficiency_skills`, `skill_to_experience_ratio`, `education_tier`, all `redrob_*`
signal fields, the raw `career_history_json` / `skills_json` / `education_json`, and `narrative_text` (headline + summary + every role
description + skill names), which feeds both embeddings and BM25.

---

## 6. Honeypot filter (`honeypot_flags.py`)

OR-logic: any single rule flags a candidate, and flagged candidates are removed before top-100 selection. All rules use only fields inside
the candidate's own record.

| Rule | Condition | Flags on the 100K pool |
|---|---|---|
| 1 | an advanced/expert skill with 0 months of use | 21 |
| 2 | at most 1.0 year of experience with 8 or more skills | 295 |
| 4 | non-current roles overlapping by more than 2 months | 0 |
| 5 | `duration_months` inconsistent with start/end dates | 0 |
| 6 | impossible education order (a higher degree ended before a lower one started) | 5,248 |

Total flagged: 5,542 (5.5%); 2,792 of the 2,969 shortlisted candidates survive. The spec expects about 80 real honeypots, so the filter is
still much broader than the target (rule 6 accounts for nearly all of it), and rules 4 and 5 are inert on this data.

**Removed rule 3.** v1.0 also flagged "tenure predates the company's inferred founding", with the founding date estimated as the ~10th
percentile of other candidates' start dates at the same company. Because the schema has no founding field this measured how early a
candidate started relative to peers, flagged 17,477 candidates (17.5% of the pool, the same rate for AI/ML titles as for everyone else,
e.g. AI Engineers at Google), and excluded 7 of the eventual top 10. It was removed on 2026-09-20. The trade-off is that the spec's named
pattern ("8 years at a company founded 3 years ago") is no longer checked.

---

## 7. JD rule layer as learned features

The JD's explicit "do not want" clauses are encoded as features and fed to the Ridge (they are not hard filters). Fitted coefficients:

| Feature | JD clause | Ridge coefficient |
|---|---|---|
| `no_recent_coding` | senior with no production code in 18+ months | -0.757 |
| `no_external_validation` | 5+ years closed source, no papers/talks/OSS | -0.612 |
| `cv_speech_robotics_no_nlp` | CV/speech/robotics without NLP/IR | -0.249 |
| `framework_enthusiast_signal` | tutorial/framework-only profile | -0.226 |
| `title_chasing_pattern` | job-hopping for titles | +0.041 |
| `is_consulting_only` | consulting-only career | **-0.003** |
| `is_pure_research` | pure research, no production | **+0.000** |
| `is_recent_wrapper_only` | <12 months of LangChain-to-OpenAI experience | **+0.000** |

**The last three carry essentially no weight**, so the model does not effectively enforce those three JD disqualifiers. Other coefficients:
`semantic_sim_to_ideal` +2.397, `behavioral_score` +1.592, `semantic_sim_to_jd` -0.865, `skill_to_experience_ratio` -0.213,
`years_of_experience` -0.170, `n_skills` +0.082, `bm25_score` +0.004, and `notice_period_penalty` +0.160 (wrong direction, neutralised; see §8).
Two representations of the JD are embedded: the literal text (`jd_embeddings[0]`) and the "ideal candidate" paragraph (`jd_embeddings[1]`),
which matches how real profiles are written better than the bullet list does.

---

## 8. Behavioural signals

**`behavioral_score`** (a feature inside the Ridge, 0-1): recency 0.15 (exponential, 90-day half-life), responsiveness 0.15, response speed
0.10, interview completion 0.15, offer acceptance 0.15 (neutral 0.5 if no history), open-to-work 0.10 (1.0 or 0.6), verification 0.10
(email/phone/LinkedIn), GitHub 0.10 (neutral 0.5 if none). Its total weight is small: response rate contributes at most about 0.24 score
points across its whole range, on a scale where the top 100 span about 4-8.

**Availability multiplier** (after the Ridge; range 0.29-1.0; `features_utils.compute_availability_multiplier`), from the JD's "a
perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is not actually available":

| Signal | Rule |
|---|---|
| Recency (reference date 2026-06-30) | no penalty up to 90 days idle, linear to x0.7 at 180 days or more |
| Recruiter response rate | no penalty at 0.4 or more (pool median 0.44), linear down to x0.5 at 0% |
| Notice period | x1.0 (<=30 d), x0.96 (<=60), x0.90 (<=90), x0.84 (>90); JD: "30+ day candidates are still in scope but the bar gets higher" |

Positive scores shrink and negative scores are pushed lower, so the modifier never lifts anyone. The parameters are judgment values taken from
the JD's wording, not fitted. The Ridge's own notice coefficient was positive (+0.16, it rewarded long notice), so `rank.py` feeds that feature
as 0 at prediction time; the trained model file is unchanged.

---

## 9. Training and validation of the ranker

`label_eval_set.py` builds `artifacts/eval_set.csv` (150 rows) **with rules, not by hand**: 30 flagged honeypots -> 0; 20 random background
candidates -> 0 (or 1 if the title looks like ML); 40 "keyword stuffers" (non-ML title, semantic similarity > 0.4) -> 1; 20 mild fits
(ML title with low similarity or consulting-only) -> 3; 40 "strong fits" (ML title, similarity > 0.4, not consulting-only) -> 5. The committed
file matches this exactly (relevance is constant within each group). `train_ranker.py` runs a 5-fold CV comparing Ridge (alpha = 1) with
LightGBM LambdaRank on the composite metric and keeps the higher-scoring model; the committed model is a Ridge.

**Consequence:** the Ridge learns to reproduce those title/similarity heuristics. Its CV score measures agreement with them, not with the
organisers' hidden relevance tiers, and there is no ground truth available to measure NDCG. The honeypot negatives came from flags that still
included the removed rule 3.

---

## 10. Online ranking step (`rank.py`)

Shortlist: FAISS top-1500 (cosine of the MiniLM embedding of each candidate's narrative against the ideal-candidate paragraph) united with BM25
top-1500 (literal JD tokens against the same narratives), giving at most 3,000 candidates (2,969 on the full pool). The shortlist is scored, the
availability multiplier applied, honeypot-flagged candidates dropped, and the top 100 written with ties broken by `candidate_id` ascending.
If the candidate file has 1,500 or fewer candidates (the small-sample sandbox case) FAISS/BM25 retrieval is skipped entirely and all of them are
scored, so the run needs neither `faiss.index` nor `bm25.pkl` (they are git-ignored and absent from the Hugging Face Space); the BM25 feature is 0 in this mode.
**Only candidates whose IDs are in the precomputed `features.parquet` can be ranked**: `rank.py` filters the features to the IDs in the candidates
file, so IDs outside the 100K pool are silently ignored, and if none match it writes an empty CSV.

---

## 11. Reasoning generation

**Default (`reasoning_grounded.py`)** — a deterministic generator, not a language model. For each finalist it reads that candidate's own
record and fills sentence frames with verified facts:

- title, employer and years; up to three JD-relevant skills the candidate actually lists, with proficiency and months (skills under 12 months of
  use or rated "beginner" are never surfaced);
- one verbatim quote from a role description that contains a JD keyword (42 of the 100 rows);
- the candidate's real concerns as noun phrases (low response rate, inactivity, notice, non-India location, years outside the JD's 5-9 band,
  no listed retrieval skill, short tenures, no open-to-work flag, no product-company experience, no described ranking evaluation, no GitHub);
- tone by rank tier (1-10 strong, 11-35 good, 36-70 reasonable, 71-100 borderline); top-10 rows show one concern, others up to two; a
  lower-ranked row with nothing to flag says the rank comes from the relative score.

`audit_reasoning()` re-checks every claim (skills with exact proficiency/months, quotes verbatim, years, employer, response rate, notice, no
first-person, no "Ranked…" opener) and the generator falls back to a minimal true sentence if any check fails. Verified 2026-09-20: 798 claims in
the 100 finalists checked against the raw `candidates.jsonl` with 0 failures; the audit caught 10 of 10 planted errors. **It is template-based:**
5 sentence-1 frames x 4 sentence-2 frames chosen by a hash of the candidate id, so some phrasing repeats across rows (for example "the JD's 5-9
year band" appears in about half the rows). 33-73 words per row.

**Legacy (`reasoning.py`, `--reasoning-mode llm`)** — a rule-based reason planner (persona, primary reason, why-not-higher, pairwise
differentiators) whose abstract labels are fed to Qwen2.5-0.5B-Instruct on CPU, with a template fallback. The prompt contains no concrete
profile facts and its validation does not check claims against the record, which is why its output was generic ("strong alignment across
semantic, technical, and readiness dimensions…"). It needs the model cached locally and roughly doubles the runtime. The NLI grounding check
mentioned in the v1.0 plan was never built.

**LLM polish, offline (`--reasoning-mode grounded-llm` builds it, `grounded-cached` uses it; `reasoning_llm_rewrite.py`)** — a local instruction model rewords each grounded
draft. The prompt carries a per-row KEEP checklist (fit wording, title, company, skills, quote, concern and availability facts), a rule that the candidate is always the subject, three worked
examples, and a **forced prefill** so the answer starts with the correct "Title at Company" phrase (this removed the model's most common error, swapping who works where). `gate_rewrite` accepts a
rewrite only if it keeps the title/company relation, at least two skills exactly as "Name (level, N mo)", every verbatim quote, every concern and availability number and the rank-tier verdict, and
adds no number, capitalised name, evaluation word ("high", "strong"), job-posting phrasing ("role requiring", "the role is located"), repeated phrase, odd capitalisation or grammar slip ("a N years",
"a N-year experience"); it also passes the record audit. Accepted rewrites are stored in `artifacts/reasoning_rewrites.json` keyed by the exact draft; at ranking time the cache is only *read* and each
entry is re-checked by the same gate, so the graded step loads no model (1 min 54 s, 4.0 GB) and any miss or failed check falls back to the grounded text. **Final v5:** 38 of the 100 finalists reworded
(31 by Qwen2.5-1.5B-Instruct, 7 by Qwen2.5-0.5B-Instruct), all 42 verbatim quotes and 3.0 skills per row kept, 0 grounding failures over 719 claims, 0 drift terms; boilerplate share of words 16.2% -> 7.7%,
distinct last-sentence openers 34 -> 47. Both models drop the quote in most rows even when told explicitly, which limits coverage. Model size mattered: at the same prompt and gate the 1.5B model produced
32 acceptable rewrites versus about 20 for the 0.5B model, at 3x the generation time (997 s vs 370 s for 100 rows), which is affordable only because generation is offline.

---

## 12. Trap defence — what actually happens

| Trap | Defence as built | Limits |
|---|---|---|
| Keyword stuffers | title/role features and similarity feed the Ridge; the eval labels explicitly include non-ML titles with high similarity as 1; the top 100 contains only AI/ML/search/recommendation/data-science titles | learned from rule-generated labels only |
| Plain-language Tier 5s | dual JD embeddings (literal + ideal paragraph); the reasoning quotes role descriptions that state the work in plain words | not measured |
| Behavioural twins | `behavioral_score` plus the availability multiplier separate similar profiles by engagement | judgment parameters |
| ~80 honeypots | five self-contained rules; none is flagged in the top 100 | the filter flags 5.5% of the pool, mostly via rule 6; real honeypot recall unknown |

---

## 13. Local validation

- `validate_submission.py` (the organisers' validator): format, 100 rows, unique ranks and IDs, non-increasing scores, tie order.
- `validate_local.py --csv … --candidates …`: the same plus a check that every ID exists in the pool and a crude tone warning for rank>70 rows
  without words like "gap" or "concern" (the grounded text hedges with "Caveat:" / "Held back by", so this can be a false positive).
- `tests/`: 8 tests for the grounded generator, 17 for the LLM-rewrite gate, prompt and cache, a no-stuffing regression test (needs `sample_submission.csv` in the repo root, which is not
  committed) and a Streamlit smoke test; 29 pass, 1 setup error in a fresh clone.
- Independent audit of the reasoning against the raw candidate file (see §11).

---

## 14. Deliverables (spec §10) — status

| Item | Status |
|---|---|
| Submission CSV | `rank.py` output; passes the validator |
| README with one reproduce command | `README.md` (needs the one-off `build_index.py` step first) |
| Full source, no hidden steps | flat repo at the root |
| Precomputed artifacts or scripts | small ones committed; `faiss.index` and `bm25.pkl` are generated by `build_index.py` (~37 min) |
| `requirements.txt` | pinned; all pins resolve |
| `submission_metadata.yaml` | present; mirrors the portal fields |
| Sandbox (<= 100 candidates, <= 5 min) | Streamlit app on a Hugging Face Space; responds, not tested end to end |
| Git history | 34 commits over 3 days before the 2026-09-20 changes, which are not yet committed |
| AI tools declaration | in `submission_metadata.yaml` |

---

## 15. Known risks

| Risk | Detail |
|---|---|
| No ground truth | labels are rule-generated; ranking quality against the hidden tiers is unmeasured |
| Near-zero weights on three JD disqualifiers | `is_consulting_only`, `is_pure_research`, `is_recent_wrapper_only` are effectively not enforced |
| Honeypot filter breadth | 5.5% of the pool flagged vs ~80 expected; rule 6 dominates |
| JD embeddings | committed `jd_embeddings.npy` differs from a fresh `build_index.py` output (max abs diff 0.095, cause unknown); a fresh reproduction may rank slightly differently |
| Tightly packed scores | ranks 50-100 span ~0.4 points, so modifier parameters move mid-list candidates a long way |
| Fewer-core grader | ~430 CPU-seconds; 2 cores need at least ~3.6 min even with perfect scaling |
| Template-based reasoning | phrasing repeats across rows; claims are fact-checked but the frames are fixed |
| Offline model cache | `all-MiniLM-L6-v2` must be in the local Hugging Face cache for an offline run |
| Pool-only ranking | candidates not in the precomputed `features.parquet` are silently ignored |

---

## 16. Changes from the v1.0 plan

- Layout is flat; there is no `precompute/` directory and no `data/job_description.json`.
- The ranker is Ridge (chosen by CV in `train_ranker.py`), trained on 150 **rule-generated** labels, not hand labels; the plan expected LightGBM
  on hand labels.
- Honeypot rule 3 and the Tier A/B company-timeline scheme were removed (§6); a rule 6 (education order) was added; `company_stats.py` is unused.
- The shortlist is at most 3,000 (2,969 in practice), not "~4-5K".
- An availability multiplier and a notice-period correction were added after the Ridge (§8).
- Reasoning is a deterministic fact-grounded generator with an audit; the plan's LLM + NLI grounding design was replaced. The earlier
  reason-planner + Qwen path is kept as `--reasoning-mode llm`.
- The plan's Docker profiling step was not done; measurements are from a 16-core machine.
