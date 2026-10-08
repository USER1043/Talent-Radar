# Reasoning Engine (`reasoning.py`)

Every finalist needs a 1–2 sentence explanation that is **grounded in real fields**, **consistent with its rank**, and **not
copy-pasted** across rows. The module separates *deciding what to say* from *saying it*.

```
top_100 rows
   │  analyze_candidate()          → CandidateAnalysis   (normalised facts)
   │  perform_decision_analysis()  → DecisionAnalysis    (why this rank, trade-offs, risk)
   │  plan_reason(prev, cur, next) → ReasonPlan          (persona, evidence, differentiators)
   ▼
 try:  batched Qwen2.5-0.5B-Instruct, few-shot, facts-only prompt
       └─ per-output cleanup + validate_explanation()
 except Exception:
       fallback_summary_generator(plan)   (deterministic, md5-seeded template variants)
```

## Stage 1 — `CandidateAnalysis`

Pulls from the scored row: `yoe`, `title`, `company`, `semantic_match` (= `semantic_sim_to_ideal`),
`behavior_score`, `notice_days`, `has_recent_coding` (= `no_recent_coding == 0`), `consulting_only`,
`open_source` / `publications` (keyword hits in narrative), `leadership` (title contains lead/staff/principal/manager/director/head/architect), and lower-cased `skills`.

## Stage 2 — `DecisionAnalysis`

Produces text fragments that explain placement:

- **Requirement coverage**: 4 JD requirements — `retrieval`, `vector`, `python`, `product ml`. Matched via skill substrings
  (e.g. `faiss/qdrant/milvus…` ⇒ vector; `not consulting_only and yoe ≥ 3` ⇒ product ml). Gives `coverage_percent` and `missing_requirements`.
- **strongest_positive**: semantic ≥ 0.58 → "Exceptional semantic match…"; else product tenure (≥ 5 yrs, non-consulting); else hands-on coding; else generic.
- **strongest_negative** (priority order): no recent coding → notice > 45 days → consulting-only → semantic < 0.54 → no external validation.
- **primary_reason**: rank ≤ 10 → "exceptional overall alignment…", else by semantic ≥ 0.56 / non-consulting / delivery-focused.
- **why_not_higher / why_not_lower**: rule ladders over coding, notice (> 30), semantic (< 0.56), OSS/publications, behaviour; and semantic (≥ 0.55), yoe (≥ 5), coding.
- **risk**: notice > 60 → "Hiring timeline constraint"; no coding → "Technical skill decay"; consulting → "Delivery-oriented culture shift".
- **recommendation**: "Shortlist" if rank ≤ 30 else "Evaluate with reservation".

## Stage 3 — `ReasonPlan`

### Persona (first matching rule wins)

| Persona | Condition (abridged) |
|---|---|
| Elite Match | yoe ≥ 5, semantic ≥ 0.58, behaviour ≥ 0.70, notice ≤ 30 |
| High Risk | no recent coding **or** notice > 90 **or** (semantic < 0.52 and notice > 60) |
| Strong Product Engineer | yoe ≥ 4, not consulting, semantic ≥ 0.55, coding |
| Senior Specialist | yoe ≥ 7, semantic ≥ 0.55 |
| Leadership Profile | leadership title and yoe ≥ 5 |
| Research-Oriented | publication keywords |
| Fast Hire | notice ≤ 15 |
| Consulting Background | consulting only |
| Emerging Candidate | yoe < 4, semantic ≥ 0.53 |
| Backend Generalist | default |

### Evidence

Up to **two** of: "strong search alignment" (semantic ≥ 0.55), "product experience", "favorable notice period",
"open source validation", "academic research profile" — default "applied engineering experience".

### Pairwise differentiators

Using the neighbours in the already-sorted list (rank *i−1*, *i+1*):

- `differentiator_beats` — why this candidate edges out the next one (notice shorter by > 15 days → semantic +0.02 → yoe +2 → active coding).
- `differentiator_loses` — why they trail the previous one (the mirrored comparisons).

This is what makes adjacent explanations *relative* rather than independent.

## Generation

### LLM path

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, loaded lazily into module globals, `device_map="cpu"`.
- All 100 prompts are built first, then `model.generate` is called **once** on the padded batch
  (`padding_side="left"`, `pad_token = eos_token`, `max_new_tokens=85`, greedy `do_sample=False`, `torch.set_num_threads(8)`).
- Prompt = system message with two few-shot examples (rank 12 "Senior Specialist", rank 98 "High Risk") + a user
  message containing **only** the plan's facts: rank, persona, primary strength, evidence, hiring constraint, why-not-higher, differentiator.
  The model never sees raw profile text — it paraphrases decisions the rules already made, which bounds hallucination.
- Post-processing: strip newlines; if the output doesn't end in `.!?`, truncate to the last sentence boundary (or append `.`).

### Deterministic fallback

If anything raises (missing weights, OOM, generation error) the whole batch falls back to
`fallback_summary_generator(plan)`: persona-specific sentence templates, three variants each, chosen by
`int(md5(candidate_id), 16) % 100 % 3`. Rank ≤ 10 uses an "unconstrained top-tier" template; rank > 20 with a
big-tech mention uses a "pedigree but placed lower" template. `plan.rank % 2 == 0` adds an optional differentiator sentence.

### `validate_explanation` — last-line guards

1. Rewrites first-person markers (`my/our/us/me` → "the", `I` → "candidate").
2. For rank ≤ 10 replaces negative phrasing ("is caution-flagged", "evaluated as a high risk candidate").
3. For rank > 50 replaces "minimal gap" with "specific technical limitations".

## Why plan first, narrate second

| Concern | How the split helps |
|---|---|
| Hallucination | LLM sees a closed set of facts, not the profile |
| Consistency with rank | Rank, persona and constraint are inputs, not inferred |
| Auditability | The `ReasonPlan` can be unit-tested and diffed without running a model |
| Latency | Single batched call; template fallback costs nothing |
| Repetition | Pairwise differentiators and hash-seeded variants break symmetry |

## Observed behaviour and caveats

- A 0.5B model paraphrases well but occasionally produces awkward claims (e.g. "excels in high-risk roles"); the explanations speak mostly in terms of rank, notice period, and a few evidence tags rather than candidate-specific facts (company, projects).
- Thresholds (0.52–0.58 for semantic similarity) are tuned to MiniLM cosine values for this JD and will not transfer to another embedding model.
- The module imports `torch` at top level, so even a template-only run pays the import cost.
- `validate_local.py` checks reasoning is non-empty and unique, and warns when rank > 70 rows lack hedging language.

Ideas to improve this stage are in [known-issues-and-roadmap.md](known-issues-and-roadmap.md).
