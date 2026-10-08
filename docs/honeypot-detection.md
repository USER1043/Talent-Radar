# Honeypot Detection

**Honeypots** are deliberately impossible profiles planted in the pool (≈80 of 100K per the challenge notes). Ground truth treats them as
irrelevant, and a top 100 containing more than 10% of them is disqualified. The pipeline therefore applies a **hard filter**, not a penalty.

Code: `honeypot_flags.py`. Output: `artifacts/honeypot_flags.parquet` (`candidate_id, is_honeypot, honeypot_reasons`).

> The organisers' spec also says honeypots need no special-casing ("a good ranking system naturally avoids them"). The filter is a safety net against the one
> hard rule (> 10% ⇒ disqualified), not the primary defence.

## OR-logic, not averaging

```python
flagged = len(reasons) > 0          # any single rule is enough
```

Averaging risk scores would let one decisive red flag be diluted by otherwise normal-looking signals. The cost is more false positives, accepted
deliberately because removing a good candidate hurts NDCG less than admitting a honeypot hurts the submission.

**Every remaining rule uses only fields inside the candidate's own record** — no cross-candidate statistics (see "Removed rule 3" below for why).

## The five rules (`check_candidate(row)`)

Numbering keeps the original design's numbers, so there is intentionally no rule 3.

| # | Rule | Condition | Constants | Flags on the 100K pool |
|---|---|---|---|---|
| 1 | Unsupported expertise | `n_zero_duration_high_proficiency_skills ≥ 1` (an *advanced/expert* skill with `duration_months == 0`) | — | 21 |
| 2 | Skill inflation | `years_of_experience ≤ 1.0` **and** `n_skills ≥ 8` | `LOW_EXPERIENCE_YEARS=1.0`, `HIGH_SKILL_COUNT=8` | 295 |
| 4 | Overlapping roles | Two **completed** roles overlap by more than 2 months | `OVERLAP_TOLERANCE_MONTHS=2` | 0 |
| 5 | Duration mismatch | A completed role whose declared `duration_months` differs from its start/end gap by > 2 months (`tenure_consistency_mismatches > 0`, computed in `parse_candidates.py`) | tolerance 2 | 0 |
| 6 | Impossible education order | A higher degree (PhD > ME/MTech/MBA/MSc > BE/BTech/BSc/BA) *ends* before a lower degree *starts* | `DEGREE_RANK` map | 5,248 |

Measured on the committed pool (from the project's architecture notes): **5,542 flagged (5.5%)**; **2,792 of the 2,969 shortlisted candidates survive** (so 177 shortlisted candidates are removed). The spec expects ≈ 80
true honeypots, so the filter is much broader than the target — rule 6 accounts for nearly all flags — and rules 4 and 5 never fire on this data.
Rules 1 and 2 are the ones that resemble the spec's own examples. No candidate in the final top 100 is flagged.

Each rule appends a human-readable string; they are stored `"; "`-joined in `honeypot_reasons`, which makes auditing easy:

```python
pd.read_parquet("artifacts/honeypot_flags.parquet").query("is_honeypot").honeypot_reasons.value_counts()
```

### Why rule 2 is a literal threshold, not a ratio

An earlier design used `n_skills / years_of_experience`. A ratio over-penalises genuinely skilled juniors (division by a small number), so the shipped rule
targets the literal trap described in the spec — *many skills with essentially no experience*.

## Removed rule 3 — "tenure predates the company's inferred founding" (a cautionary tale)

The original design flagged a candidate whose role started more than 6 months before an *inferred* company founding date. The candidate schema has no
founding-date field, so the date was estimated from the data itself: the ~10th-percentile start date across all career entries mentioning that company
(`company_stats.py`, trusted only for companies with ≥ 5 mentions — "Tier A"; rarer companies were skipped — "Tier B").

Measured behaviour (2026-09-20, from the project's notes):

- It flagged **17,477 candidates (17.5% of the pool)**, at the **same rate for AI/ML titles as for everyone else** (e.g. AI Engineers at Google).
- It excluded **7 of the eventual top 10**.
- Reason: with no true founding dates, the statistic measured *how early a candidate started relative to peers*, not whether the company existed.

It was therefore removed. `company_stats.py` is **kept for reference only** and is no longer run; `honeypot_flags.py` no longer takes `--company-stats`.
**Trade-off:** the spec's named pattern ("8 years at a company founded 3 years ago") is no longer checked at all.

Lessons worth keeping if you propose a replacement: validate any new rule by (a) its flag rate overall, (b) whether the flag rate differs between target-role titles and
others, and (c) how many current top-ranked candidates it would eject — a rule that flags everyone equally is noise, not a honeypot detector.

## Where the filter is applied

`rank.py` stage 7 — after scoring, on the shortlist only:

```python
shortlist_df = pd.merge(shortlist_df, df_hp, on="candidate_id")
non_hp_df = shortlist_df[shortlist_df["is_honeypot"] == False]
```

Flagged candidates can still consume retrieval slots but never reach the top 100. The flag itself is **not** a ranker feature. Honeypots were used as relevance-0 examples in the
eval set (30 of 150 rows) — note those eval honeypots were drawn from flags that **still included the removed rule 3**, and `eval_set.csv` / `ranker_model.pkl` have not been regenerated since.

## Performance note

`honeypot_flags.main()` uses `DataFrame.iterrows()` over 100K rows with a `json.loads` per row: ~5 s offline. Vectorise before moving it into the timed path.

## Extending the rules

1. Add the check to `check_candidate` and append a descriptive reason string.
2. If it needs a new aggregate, compute it in `parse_candidates.flatten()` so the parquet carries it, rather than re-deriving it from JSON.
3. Re-run `python honeypot_flags.py --features artifacts/features.parquet --out artifacts/honeypot_flags.parquet` and compare per-rule flag counts and the overall rate with the table above.
4. Check that the rate does not differ suspiciously between AI/ML titles and others, and how many shortlisted/top-ranked candidates it removes.
5. If the sampled eval honeypots change, regenerate `eval_set.csv` and retrain (`label_eval_set.py`, `train_ranker.py` — both **overwrite** committed artifacts).
