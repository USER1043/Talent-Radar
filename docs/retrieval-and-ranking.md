# Retrieval & Ranking

This is the heart of the system: how 100K profiles become 100 ordered ones.

## 1. Two representations of the JD

`build_index.extract_jd_texts(job_description.docx)` returns two strings:

1. **Literal JD** — every non-empty paragraph.
2. **Ideal-candidate paragraph** — the section that starts at the paragraph containing
   both "ideal candidate" and "imagining" and stops at the paragraph containing "disqualifiers".

Both are embedded with `all-MiniLM-L6-v2` and saved in `jd_embeddings.npy`
(`[0]` literal, `[1]` ideal). Rationale: bullet-style JD text matches bullet-style
skill lists (and keyword stuffers), whereas the narrative "ideal candidate" text
resembles how a *real* profile summary reads. This helps "plain-language" strong
candidates who never write "RAG" or "Pinecone" but describe having built a
recommender at a product company.

> If `python-docx` is missing, `extract_jd_texts` silently returns a hard-coded
> two-sentence fallback JD. Retrieval then still runs but quality collapses — install the dependency.

## 2. Hybrid retrieval (shortlist)

| Leg | Index | Query | Top-K |
|---|---|---|---|
| Semantic | `faiss.IndexFlatIP(384)` over L2-normalised MiniLM vectors (inner product = cosine) | **ideal-candidate** embedding | 1500 |
| Lexical | `BM25Okapi` over `\w+` tokens | **literal JD** tokens | 1500 |

The shortlist is the **union** (≈ 3K candidates in the committed run; the design spec
budgeted 4–5K). Why both?

- Dense retrieval catches paraphrase / plain-language fits; it can miss rare exact terms.
- BM25 catches exact skill/tool vocabulary; it misses synonyms.
- Their failure modes differ, so the union raises the recall ceiling *before* the learned ranker sees anyone.

Retrieval is deliberately **exact** (`IndexFlat*`), not HNSW/IVF — 100K×384 is tiny, and it removes ANN recall as a variable.

## 3. Structured JD-rule features (`features_utils.py`)

The JD has an explicit "things we do NOT want" list. Each clause becomes a 0/1
feature computed from skills, title, narrative text, or tenure. They are *soft*
signals — inputs to the learned model — not hard filters.

| Feature | Fires (=1.0) when | JD clause |
|---|---|---|
| `is_pure_research` | narrative has a research term (`research lab`, `phd`, `thesis`, `publications`, `postdoc`, …) **and no** production term (`production`, `deployed`, `scale`, `kubernetes`, …) | Pure research, no production → "will not move forward" |
| `is_recent_wrapper_only` | has LangChain/LlamaIndex/OpenAI/prompt-engineering skill, **no** classic ML skill, and `yoe < 1.5` | "<12 months of LangChain-calling-OpenAI" |
| `no_recent_coding` | current **title** contains PM / HR / recruiter / sales / marketing / … | Proxy for "hasn't written production code" and for keyword-stuffers |
| `cv_speech_robotics_no_nlp` | has CV/speech/robotics skill and no NLP/IR skill | CV/speech/robotics without NLP/IR |
| `no_external_validation` | `yoe > 5` and none of paper / OSS / github / talk / conference / patent in narrative | 5+ years closed-source |
| `title_chasing_pattern` | `avg_tenure_months < 18` and `n_employers ≥ 2` | Title-chasers |
| `framework_enthusiast_signal` | ≥1 framework (LangChain, LlamaIndex, Haystack, Flowise, AutoGen) and no core skill (Python, PyTorch, TF, sklearn, numpy, pandas, …) | Framework enthusiasts |
| `is_consulting_only` (from parquet) | *every* career entry is in `CONSULTING_FIRMS` (TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant, Capgemini) | Consulting-only careers |

Other numeric inputs: `years_of_experience`, `n_skills`, `skill_to_experience_ratio`.

All of these are substring matches over lower-cased text. They are intentionally
cheap and noisy; see the trade-offs in [design-decisions.md](design-decisions.md).

### Notice-period penalty (`compute_notice_period_penalty`)

| Notice days | Penalty |
|---|---|
| NaN or ≤ 30 | 0.0 |
| 31–60 | 0.2 |
| 61–90 | 0.5 |
| > 90 | 0.8 |

The JD says sub-30-day is ideal, 30+ is "still in scope but the bar gets higher". The design doc asked for a smooth
curve; the implementation is a step function.

### Behavioural score (`compute_behavioral_score`) — range 0–1

A weighted sum of eight terms (weights sum to 1.0), with "today" pinned at **2026-06-30** so runs are reproducible:

| Term | Weight | Definition |
|---|---|---|
| recency | 0.15 | `exp(−ln2 · days_since_last_active / 90)` (90-day half-life); default 180 days if date is unparseable |
| responsiveness | 0.15 | `recruiter_response_rate` (default 0.5 if null) |
| speed | 0.10 | `1 / (1 + avg_response_time_hours / 24)` (0.5 if null/negative) |
| reliability | 0.15 | `interview_completion_rate` (0.5 if null) |
| offer signal | 0.15 | `offer_acceptance_rate` if ≥ 0 else 0.5 (−1 sentinel = no history) |
| availability | 0.10 | 1.0 if `open_to_work_flag` else 0.6 |
| trust | 0.10 | 0.5·verified_email + 0.3·verified_phone + 0.2·linkedin_connected |
| github | 0.10 | `github_activity_score / 100`, 0.5 if −1/null |

The JD explicitly says a "perfect on paper" candidate who hasn't logged in for six months and has a 5% response rate is
effectively unavailable — this score is the answer to that. Weights are hand-set; the ranker only sees the single
composite (not its components).

## 4. Candidate-level similarity features (online)

For each shortlisted candidate, `rank.py` re-embeds `narrative_text` (instead of reading the 100K-row matrix) and computes:

- `semantic_sim_to_jd` = cos(candidate, literal JD)
- `semantic_sim_to_ideal` = cos(candidate, ideal paragraph)
- `bm25_score` = BM25 score of the literal JD tokens against the candidate (looked up from the full-pool score vector)

## 5. The learned ranker (`train_ranker.py`)

**Problem:** no ground-truth relevance exists, so hand-picked weights over 16 features would be guesses.

**Approach:** build a 150-row labelled eval set ([evaluation-and-testing.md](evaluation-and-testing.md)),
then compare two models by **5-fold CV on the composite hackathon score**:

| Model | Config |
|---|---|
| Ridge regression | `alpha=1.0`, regress relevance (0/1/3/5) on the 16 features |
| LightGBM LambdaRank | `n_estimators=50, learning_rate=0.05, max_depth=3, num_leaves=7`, whole fold treated as one query group |

The winner (`lgb ≥ ridge` ⇒ LightGBM, else Ridge) is refit on all 150 rows and pickled with its
feature names. **Ridge won**, which is what is committed. At ~150 examples a 16-feature linear model with L2 shrinkage is
less prone to overfitting than boosted trees even with shallow trees.

Metric helpers in the script:

- `ndcg_k(relevances, k)` — standard DCG/IDCG with `log2(i+2)`.
- `average_precision` and `precision_at_10` — binarise at `relevance ≥ 3`.
- `compute_composite_score` — `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.

## 6. Final ordering

```python
non_hp_df.sort_values(by=["score", "candidate_id_num"], ascending=[False, True]).head(100)
```

The numeric part of the ID is extracted so the tie-break is numeric-ascending, which matches the validator's
"candidate_id ascending on equal scores" rule for fixed-width IDs.

## Trap handling summary

| Trap | Defence |
|---|---|
| Keyword stuffers (HR Manager with AI skills) | `no_recent_coding` (title coherence) + relevance label 1 in training + embeddings of the *whole* narrative rather than skill list |
| Plain-language strong candidates | Ideal-paragraph embedding + dense leg of retrieval |
| Behavioural twins (same skills, one reachable) | `behavioral_score`, `notice_period_penalty` |
| Honeypots | Separate hard filter — see [honeypot-detection.md](honeypot-detection.md) |
