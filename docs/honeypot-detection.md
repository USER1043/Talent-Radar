# Honeypot Detection

**Honeypots** are deliberately impossible profiles planted in the pool (≈80 of 100K per the
challenge notes). Ground truth treats them as irrelevant, and a top-100 containing more than
10% of them is disqualified. They therefore get a **hard filter**, not a penalty.

Code: `company_stats.py` (statistics) and `honeypot_flags.py` (rules). Output:
`artifacts/honeypot_flags.parquet`.

## OR-logic, not averaging

```python
flagged = len(reasons) > 0          # any single rule is enough
```

Averaging risk scores would let one decisive red flag be diluted by otherwise normal-looking signals — which is
exactly how an adversarial profile would slip through. The cost is more false positives, accepted deliberately
because removing one good candidate hurts NDCG less than admitting a honeypot hurts the submission.

## The six rules (`check_candidate`)

| # | Rule | Condition | Constants |
|---|---|---|---|
| 1 | Unsupported expertise | `n_zero_duration_high_proficiency_skills ≥ 1` (an *advanced/expert* skill with `duration_months == 0`) | — |
| 2 | Skill inflation | `years_of_experience ≤ 1.0` **and** `n_skills ≥ 8` | `LOW_EXPERIENCE_YEARS=1.0`, `HIGH_SKILL_COUNT=8` |
| 3 | Predates company | A role's `start_date` is more than 6 months **before** the company's `earliest_plausible_date`, for corroborated companies only | `COMPANY_FOUNDING_BUFFER_MONTHS=6` |
| 4 | Overlapping roles | Two **completed** roles overlap by more than 2 months | `OVERLAP_TOLERANCE_MONTHS=2` |
| 5 | Duration mismatch | Any completed role where declared `duration_months` differs from its start/end gap by > 2 months (`tenure_consistency_mismatches > 0`, computed in `parse_candidates`) | tolerance 2 |
| 6 | Impossible education order | A higher degree (PhD > ME/MTech/MBA/MSc > BE/BTech/BSc/BA) *ends* before a lower degree *starts* | `DEGREE_RANK` map |

Each rule appends a human-readable string; they are stored `"; "`-joined in `honeypot_reasons`,
which makes spot-checking flagged profiles easy:

```python
pd.read_parquet("artifacts/honeypot_flags.parquet").query("is_honeypot").honeypot_reasons.value_counts()
```

### Why rule 2 is a literal threshold, not a ratio

An earlier design used `n_skills / years_of_experience`. A ratio over-penalises genuinely
skilled juniors (division by a small number), so the shipped rule targets the literal trap
described in the spec — *many skills with essentially no experience*.

## Company-timeline statistics (`company_stats.py`)

Rule 3 needs a notion of when a company "could have existed". The data has no founding years, so
it is **inferred from the pool itself**: for each company string, collect every start date across all career entries.

| Tier | Condition | Treatment |
|---|---|---|
| **A — corroborated** | ≥ 5 mentions | `earliest_plausible_date` = value at index `max(0, int(0.10·(n−1)))` of the sorted start dates (≈ 10th percentile) |
| **B — rare** | 1–4 mentions | Statistic is stored but `is_corroborated=False`; rule 3 is **skipped**. These candidates are still covered by rules 1, 2, 4, 5, 6 |

Why a **low percentile** rather than the median? With very few mentions the median is just the midpoint, so about half of
any small group would be guaranteed to fall "before" it — a coin flip, not evidence. A low percentile over a large
enough *n* approximates a lower bound, and the ≥ 5 requirement avoids trusting a percentile of two points.

## Where the filter is applied

`rank.py` stage 7 — after scoring, on the shortlist only:

```python
shortlist_df = pd.merge(shortlist_df, df_hp, on="candidate_id")
non_hp_df = shortlist_df[shortlist_df["is_honeypot"] == False]
```

Because it runs after scoring, flagged candidates can still consume retrieval slots but never reach the top 100.
They are also used in the eval set as explicit relevance-0 examples (30 of the 150 rows), though the flag
itself is *not* a ranker feature — the ranker never learns to avoid honeypots; the filter does that job.

## Performance note

`honeypot_flags.main()` uses `DataFrame.iterrows()` over 100K rows with a `json.loads` per row. That is fine as an
offline step (seconds to low minutes) but must be vectorised before being moved into the timed path.

## Extending the rules

1. Add the check to `check_candidate` and append a descriptive reason string.
2. If it needs a new aggregate, compute it in `parse_candidates.flatten()` (so the parquet carries it) rather than re-deriving it from JSON.
3. Re-run `honeypot_flags.py`, then inspect `honeypot_reasons` counts to see how much the new rule changes the flag rate.
4. Re-run `label_eval_set.py` / `train_ranker.py` if the change alters which rows are sampled as honeypots.

Open question tracked in [known-issues-and-roadmap.md](known-issues-and-roadmap.md): how many shortlist
candidates the rules remove versus the ≈80 expected honeypots.
