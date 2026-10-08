# Data Model & Artifacts

## Input: one candidate record (JSONL line)

Fields as consumed by `parse_candidates.flatten()`:

```jsonc
{
  "candidate_id": "CAND_0068351",              // regex ^CAND_\d{7}$
  "profile": {
    "headline": "...", "summary": "...",
    "current_title": "...", "current_company": "...",
    "current_company_size": "...", "current_industry": "...",
    "location": "...", "country": "...",
    "years_of_experience": 6.5
  },
  "career_history": [
    { "company": "...", "title": "...", "description": "...",
      "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD|null",
      "is_current": false, "duration_months": 30 }
  ],
  "skills": [ { "name": "FAISS", "proficiency": "expert|advanced|...", "duration_months": 24 } ],
  "education": [ { "degree": "B.Tech", "tier": "...", "start_year": 2012, "end_year": 2016 } ],
  "redrob_signals": { /* 23 behavioural fields, see below */ }
}
```

Nulls are normal (e.g. `end_date` for current roles) and are handled by `schema_utils.parse_date`, which returns `None` instead of raising.

### The 23 `redrob_signals`

Flattened to columns `redrob_<name>`:

`profile_completeness_score, signup_date, last_active_date, open_to_work_flag,
profile_views_received_30d, applications_submitted_30d, recruiter_response_rate,
avg_response_time_hours, connection_count, endorsements_received, notice_period_days,
preferred_work_mode, willing_to_relocate, github_activity_score, search_appearance_30d,
saved_by_recruiters_30d, interview_completion_rate, offer_acceptance_rate,
verified_email, verified_phone, linkedin_connected`
plus two structured ones: `expected_salary_range_inr_lpa` → `redrob_expected_salary_min/max`, and
`skill_assessment_scores` → `redrob_skill_assessment_scores_json`.

**Sentinel values:** `-1` means "no data" for `offer_acceptance_rate` and `github_activity_score`
(no offer history / no GitHub linked). `compute_behavioral_score` maps these to a neutral 0.5 rather than penalising.

## `artifacts/features.parquet` — the flat feature table

One row per candidate, written in 5,000-row batches by a streaming `ParquetWriter`
(never holds all 100K raw records in memory).

| Column group | Columns |
|---|---|
| Identity / profile | `candidate_id, years_of_experience, current_title, current_company, current_company_size, current_industry, location, country` |
| Career aggregates | `n_career_entries, n_employers, total_tenure_months, avg_tenure_months` (= total / distinct employers) |
| Integrity | `tenure_consistency_mismatches` (completed roles where `|date-gap − duration_months| > 2`) |
| JD-relevant flags | `is_consulting_only, has_product_company_experience` |
| Skills | `n_skills, n_expert_or_advanced_skills, n_zero_duration_high_proficiency_skills, skill_to_experience_ratio` (= `n_skills / max(yoe, 0.5)`) |
| Education | `education_tier` (first entry) |
| Text | `narrative_text` = headline + summary + every career description + skill names |
| Raw nested (JSON strings) | `career_history_json, skills_json, education_json, redrob_skill_assessment_scores_json` |
| Signals | `redrob_*` (23 fields, see above) |

### Why nested data is stored as JSON strings

Expanding career/skills into columns would create hundreds of mostly-empty columns
and make batch schemas unstable. Downstream consumers (`company_stats`, `honeypot_flags`,
`features_utils`) `json.loads` the one column they need. Trade-off: parsing cost on every use
and no columnar predicate pushdown — acceptable here because those consumers are offline or small-N.

### First-person sanitising

`sanitize_first_person` rewrites "I am / I have / my experience…" to third-person
("The candidate is …") in headline, summary and career descriptions **before**
`narrative_text` is built. Reason: both the embeddings and the LLM prompts consume
this text, and first-person leakage produced reasoning that read as if the candidate were speaking.

> Gotcha: the schema is inferred per batch. If a nullable column is entirely `None`
> in one batch, pyarrow may infer a different type than in earlier batches and `write_table`
> will fail. The code comments flag the fix (pass an explicit `pa.schema`).

## `artifacts/company_stats.parquet`

| Column | Meaning |
|---|---|
| `company` | Exact stripped company string |
| `occurrence_count` | Number of career entries across the pool mentioning it |
| `is_corroborated` | `occurrence_count >= 5` (Tier A) |
| `earliest_plausible_date` | ≈ 10th-percentile start date across mentions |

## `artifacts/honeypot_flags.parquet`

`candidate_id, is_honeypot (bool), honeypot_reasons ("; "-joined rule messages)`.

## Retrieval artifacts

| File | Produced by | Content | Committed? |
|---|---|---|---|
| `faiss.index` | `build_index.py` | `IndexFlatIP(384)` over L2-normalised MiniLM embeddings of `narrative_text`; row *i* = row *i* of `features.parquet` | ❌ (gitignored, ~150 MB) |
| `bm25.pkl` | `build_index.py` | Pickled `BM25Okapi` over `\w+` lower-cased tokens of `narrative_text` | ❌ (gitignored, ~185 MB) |
| `jd_embeddings.npy` | `build_index.py` | shape (2, 384): `[0]` literal JD text, `[1]` "ideal candidate" paragraph | ✅ |

## Training artifacts

| File | Content |
|---|---|
| `eval_set.csv` | 150 rows: `candidate_id, relevance, sample_type, current_title, current_company, years_of_experience, is_honeypot, is_consulting_only, semantic_sim, skills` |
| `ranker_model.pkl` | `{"type": "ridge"\|"lightgbm", "model": <fitted>, "feature_names": [...]}` — currently **Ridge** |

## Output: `submission.csv`

`candidate_id, rank (1–100), score (float, non-increasing), reasoning (1–2 sentences)`.
`score` is the raw ranker output (not normalised to 0–1); the top rows are typically
well separated from the pack (≈ 8.2 / 7.8 / 7.7 then ≈ 4.3–5.5 in the committed file).

## The 16 ranker features

Order is whatever `X_df.columns` was at training time and is stored in the pickle:

| # | Feature | Source |
|---|---|---|
| 1 | `years_of_experience` | parquet |
| 2 | `n_skills` | parquet |
| 3 | `is_consulting_only` | parquet → 0/1 |
| 4 | `is_pure_research` | text rule on `narrative_text` |
| 5 | `is_recent_wrapper_only` | skills + yoe rule |
| 6 | `no_recent_coding` | title rule |
| 7 | `cv_speech_robotics_no_nlp` | skills rule |
| 8 | `no_external_validation` | yoe + text rule |
| 9 | `title_chasing_pattern` | `avg_tenure_months`, `n_employers` |
| 10 | `framework_enthusiast_signal` | skills rule |
| 11 | `behavioral_score` | 8-signal weighted sum |
| 12 | `notice_period_penalty` | step function |
| 13 | `skill_to_experience_ratio` | parquet |
| 14 | `semantic_sim_to_jd` | cos(narrative, literal JD) |
| 15 | `semantic_sim_to_ideal` | cos(narrative, ideal-candidate paragraph) |
| 16 | `bm25_score` | BM25 of literal JD vs narrative |

Rules and thresholds for 4–12 are in [retrieval-and-ranking.md](retrieval-and-ranking.md).
