# Reasoning Engine

Every finalist needs a 1–2 sentence explanation that is **grounded in the candidate's own record**, **consistent with its rank**, and
**not copy-pasted** across rows. The project has gone through three generations of this stage; all three are still in the repo and
selectable with `rank.py --reasoning-mode`.

| Mode | Module | LLM at ranking time? | Needs | Status |
|---|---|---|---|---|
| `grounded` (**default**) | `reasoning_grounded.py` | No | nothing | Deterministic frames filled with audited facts |
| `grounded-cached` | `reasoning_llm_rewrite.py` | No | `artifacts/reasoning_rewrites.json` | Uses LLM rewrites precomputed offline, each re-checked by a strict gate; otherwise the grounded text |
| `slotted-cached` | `reasoning_slotted.py` | No | cache entries with a `"template"` | LLM wrote only connecting words; code re-fills the facts. **The committed cache has no template entries**, so every row falls back to the grounded text |
| `grounded-llm` | `reasoning_llm_rewrite.py` | **Offline build step** | local Qwen weights | Rewrites drafts with a local model and writes accepted ones to the cache (slow: ~6 min for 100 rows with 0.5B, ~17 min with 1.5B) |
| `llm` | `reasoning.py` | Yes (Qwen2.5-0.5B) | local Qwen weights | **Legacy** reason-planner + generation; ~2 extra minutes, ~3 GB extra RAM |

The idea throughout: **the language model is never the source of facts.** Facts come from the record; an audit or gate verifies
every claim before text is accepted; the worst case is always a plain, true, deterministic sentence.

---

## 1. Default: `reasoning_grounded.py`

### Pipeline

```
row ──build_facts──► Facts ──compose_reasoning(f, rank)──► text ──audit_reasoning──► ok? ──► use
                                                                         └─ problems ─► minimal true sentence
```

### `build_facts(row)` → `Facts`

Everything is read from the candidate's row:

| Fact | How it is obtained |
|---|---|
| title, company, years | `current_title`, `current_company`, `years_of_experience` |
| **skills_named** (up to 3) | From `skills_json`, only skills in the JD skill groups below, with ≥ 12 months of use and not "beginner". One skill per group first (so three vector databases don't crowd out ranking skills), then fill by weight → proficiency → months |
| has_core_skill | any chosen-eligible skill in a weight-3 group |
| **quote** | Best role-description sentence matching JD keywords (retriev/ranking/search/recommend/embedding/vector/relevance/evaluat/ndcg/a-b/index), current role first, **verbatim**, ≤ 16 words, cut at a clause boundary, never leaving an open parenthesis |
| product_ok | `has_product_company_experience and not is_consulting_only` |
| resp / notice / idle_days / open_to_work / github / location / relocate | `redrob_*` signals; idle days computed against `AS_OF_DATE` (2026-06-30) |
| n_employers, avg_tenure_months | parquet columns |
| eval_mentioned | narrative mentions evaluation words (evaluat, NDCG, MRR, A/B, offline/online metric) |

JD skill groups and weights (`_GROUPS`): **search** (3: Semantic Search, Vector Search, Information Retrieval[ Systems], Search Infrastructure, BM25, Indexing Algorithms),
**embeddings** (3), **vector_infra** (3: FAISS, Qdrant, Weaviate, Milvus, Pinecone, pgvector, Elasticsearch, OpenSearch, Haystack), **ranking** (3: Learning to Rank, Recommendation Systems),
**llm_ft** (1), **python** (1). Only skills the candidate actually lists are ever named.

### `compose_reasoning(f, rank)`

- **Tier by rank:** 1–10 → strong, 11–35 → good, 36–70 → reasonable, 71–100 → borderline. Each tier has three verdict phrases
  ("Strong fit", "Clear fit", "Very good match"; … "Borderline fit", "Partial match", "Marginal match"), chosen by `md5(candidate_id)`.
- **Sentence 1** (5 frames, hash-chosen): verdict + `Title at Company` + years + up to two positives, e.g.
  `6.4-year Lead AI Engineer at Sarvam AI (very good match), with hands-on skills including Qdrant (expert, 78 mo), … and product-company experience.`
  Positives are the skills clause plus one rotating fact (verbatim quote / product-company experience / "experience inside the JD's 5-9 year band"); order may flip.
- **Sentence 2** (4 frames): the candidate's real **concerns** as noun phrases — at most 1 for top-10 rows, 2 otherwise — or, if none, an
  availability summary (tiers 1–2) or an honest "no single disqualifying gap, but a lower relative score…" note (tiers 3–4).

Concerns (in priority order): response rate < 30%; ≥ 150 idle days; notice ≥ 90 days; non-India base (with relocation stance); years < 5 or > 9.5;
no core retrieval skill; ≥ 4 employers averaging < 20 months; no open-to-work flag; notice 60–89 days; no product-company experience; no described
ranking-evaluation work; no linked GitHub.

### `audit_reasoning(text, row)` → list of problems

Re-checks every claim against the record:

1. Every JD skill name in the text exists in the candidate's skills, and any `(proficiency, N mo)` matches exactly (title/company/quotes are blanked first so words inside them don't trigger false hits; longer skill names are matched first).
2. Every double-quoted span is a verbatim substring of the role descriptions (ellipsis tolerated).
3. Any `Name (level, N mo)` claim names a real listed skill, even outside the JD groups.
4. Title and company are present; every "N years / N-year" mention equals the candidate's years.
5. Response-rate percentages and `N-day notice` match the record; "immediate availability" only when notice is 0.
6. No first-person pronoun outside quotes; no generic "Ranked …" opener.

`generate_reasonings_grounded` falls back to `"{title} at {company} with {yrs} years of experience; details on the profile were not specific enough to justify more."` if an audit ever fails (reported as "should never happen").

### Properties

- **Deterministic**: same record + rank ⇒ same text. **Fast**: ~0.1 s for 100 rows.
- **Template-based**: 5 × 4 frames, so some phrasing repeats across rows (e.g. "the JD's 5-9 year band"); claims are fact-checked, wording is not rich.
- Reported verification (project README/spec): 798 claims in the 100 finalists checked against the raw `candidates.jsonl`, 0 failures; the audit caught 10 of 10 planted errors.

---

## 2. Offline LLM polish: `reasoning_llm_rewrite.py`

Goal: more natural prose without giving up the guarantees. The LLM runs **once, offline**, and ranking only *reads* the result.

```
OFFLINE  (--reasoning-mode grounded-llm)                       RANKING  (--reasoning-mode grounded-cached)
draft = compose_reasoning(...)                                  draft = compose_reasoning(...)
messages = system rules + 3 worked examples                     entry = cache[sha1(draft)]
           + "DRAFT / KEEP / QUOTE RULE"                        problems = gate_rewrite(entry.rewrite, draft, row, rank)
prefill = "Title at Company" phrase (forced start)              ok? use rewrite : use draft
generate (greedy) → gate_rewrite → accept
   └─ fail → one repair turn with the gate's complaints → gate → else keep draft
accepted ⇒ cache[sha1(draft)] = {rewrite, model, attempt}  →  artifacts/reasoning_rewrites.json
```

Key mechanisms:

- **KEEP checklist** per row (`keep_list`): verdict wording, title, company, years, exact skill claims, quotes, concern/availability phrases.
- **Forced prefill** (`prefill_phrase`): the answer must start with one of four correct "Title at Company" phrasings, chosen by candidate number. This removed the model's most common error — swapping who works where.
- **`gate_rewrite`** accepts only if all hold (selected checks): 25–95 words, ≤ 3 sentences, no list output, no rank mention, no first person, no job-posting drift words ("recruiting", "offers", "requires", "proven", "highly"…) that the draft didn't already use, no new retained-concern keyword, no added evaluation word ("high/strong/solid/…"), no odd capitalisation or "a N years" grammar slip, no repeated 5-gram, title+company present *and* in a valid relation, skill triples verified against the record, ≥ 2 of the draft's skills kept exactly, every verbatim quote kept (and none invented), no number added or required number dropped, verdict wording preserved, no new capitalised names, tone consistent with the tier (no weak-fit wording in the top 10, no strong-fit wording at rank > 70), and the same `audit_reasoning` pass.
- **Cache keyed by the exact draft** (`sha1`). Because the key is the draft text, any change to the draft generator or the candidate's rank/record naturally invalidates entries (cache miss → draft).
- **Re-gating at ranking time**: a cached rewrite is trusted only after passing the gate again for *this* row and rank.

Reported result (project README): 38 of 100 finalists reworded (31 by Qwen2.5-1.5B-Instruct, 7 by Qwen2.5-0.5B-Instruct), all 42 quotes and ~3.0 skills/row kept, 0 grounding failures over 719 claims, boilerplate share of words 16.2% → 7.7%.
The committed cache file holds 39 entries (32 from 1.5B, 7 from 0.5B). Both models often drop the verbatim quote even when told to keep it, which caps coverage — rewrites that lose it are simply rejected.

Practical notes: set `LLM_DTYPE=bfloat16` to halve model memory; pass `--reasoning-diagnostics path.json` to get a per-row accept/fallback report; model weights must be in the local Hugging Face cache.

---

## 3. Slotted templates: `reasoning_slotted.py`

A stricter variant: the LLM writes only the **connecting words**, with placeholders such as `[WHO]`, `[YEARS]`, `[FIT]`, `[SKILLS]`, `[QUOTE]`, `[PRODUCT]`, `[CONCERN1]`, `[CONCERN2]`, `[AVAIL]`, `[NOTE]`. Code fills each slot with text produced by `build_slot_parts` (from the record), so the model cannot drop, reword or invent a fact.

- `gate_slotted` checks the placeholders (all required present exactly once, none unknown), and that the words *outside* slots are harmless: no digits, quotes, names, first-person, rank/position talk, job-posting drift, evaluation/praise words, nothing beyond plain connectives plus the row's own draft vocabulary; concern slots framed as concerns and positive facts never placed in concern sentences; articles/prepositions around slots sane; and the **filled** text passes length, subject, repetition and record audits.
- Offline generation (`generate_reasonings_slotted`) makes up to 4 attempts (attempt 1 greedy, later ones sampled), keeps every accepted template, and prefers the **least-reused skeleton** to maximise diversity (`MAX_SAME_SKELETON = 2`).
- The cache stores the *template*; `generate_reasonings_slotted_cached` re-gates and re-fills it from the current record.
- **Not wired to a CLI mode for building**: the module docstring mentions `slotted-llm`, but `rank.py` only exposes `slotted-cached` (read). `generate_reasonings_slotted` must be called from Python to populate templates, and the committed cache contains none.

---

## 4. Legacy: `reasoning.py` (`--reasoning-mode llm`)

Kept for reference and comparison.

```
analyze_candidate → CandidateAnalysis → perform_decision_analysis → DecisionAnalysis → plan_reason → ReasonPlan
   → one batched Qwen2.5-0.5B call (few-shot, facts-only prompt)  ── on exception ──► fallback_summary_generator (md5-seeded templates)
```

- **Planner**: persona rules (Elite Match, High Risk, Strong Product Engineer, Senior Specialist, Leadership Profile, Research-Oriented, Fast Hire, Consulting Background, Emerging Candidate, Backend Generalist), a JD-requirement coverage count, strongest positive/negative, "why not higher / lower", risk, and pairwise differentiators versus the neighbours at rank *i−1* / *i+1*.
- **Prompt**: only abstract labels from the plan (rank, persona, primary strength, evidence tags, constraint, limit reason, differentiator) — **no concrete profile facts**.
- **Generation**: all prompts in one `generate` call (`max_new_tokens=85`, greedy, left padding, `torch.set_num_threads(8)`); output trimmed to a sentence boundary; `validate_explanation` swaps a few phrases and first-person markers.
- **Why it was replaced as the default**: with no concrete facts in the prompt and no claim checking, output was generic ("strong alignment across semantic, technical, and readiness dimensions…"), occasionally awkward ("excels in high-risk roles"), and it needs the model cached locally and roughly doubles runtime. The NLI grounding check proposed in the original plan was never built.
- `torch` and `transformers` are now imported **lazily** only on this path (rank.py imports `reasoning` inside the `llm` branch).

---

## 5. Testing the reasoning stack

| File | Focus |
|---|---|
| `tests/test_reasoning_grounded.py` (8) | audit passes and text is specific; only owned skills named; concerns real and specific; no-concern case is honest; tone follows rank; determinism; many candidates all distinct and grounded; audit catches fabrication |
| `tests/test_reasoning_llm_rewrite.py` (17) | gate accepts faithful rewrites and rejects each kind of unfaithful one (dropped skill/quote, altered numbers, drift wording, claims the draft never made, wrong tone for tier, missing verdict, grammar slips); keep-list and prefill correctness; fallback when the generator crashes; repair pass; cache round-trip and "cached mode uses only gate-verified rewrites"; live mode writes the cache |
| `tests/test_reasoning_slotted.py` | slot parts are verified facts; missing/duplicate/unknown placeholders rejected; model can't write facts; praise/posting wording rejected; plain-connective vocabulary; concern framing; article repair; wrong subject; cached template is re-filled from the *current* record; diversity picker |

All LLM-dependent paths take an injectable `generator` function, so they are tested without loading a model.

## 6. Extending it

- **New concern or fact** → add to `build_facts` / `_concerns`, extend `audit_reasoning` so the new claim is checkable, add a test. Remember that changing draft text invalidates cache keys (re-run `grounded-llm` to rebuild).
- **More natural wording** → grow the cache (`grounded-llm` with a larger local model) or finish a CLI path for slotted generation; keep the gate as the only door into the output.
- **Different embedding model** → thresholds elsewhere (not here) change; this stage does not depend on similarity scores.
