# Design Decisions

Each entry: **Context → Decision → Alternatives considered → Trade-off / what would change it.**
Contributors should read the relevant entry before changing the behaviour it describes.

---

## D1. Offline precompute / online rank split

- **Context:** graded run is ≤ 5 min, CPU-only, no network, ≤ 16 GB; 100K profiles.
- **Decision:** parse, embed, index, flag honeypots and train the ranker offline; online only loads artifacts, embeds a ~3K shortlist and runs a linear model.
- **Alternatives:** score all 100K online (impossible in budget); precompute everything including final scores (but the JD would be baked in and `rank.py` would be a lookup, not a ranking system).
- **Trade-off:** reproducibility now depends on large artifacts that are partly gitignored; a stale artifact silently changes results.

## D2. Hybrid retrieval (FAISS ∪ BM25) with a wide shortlist

- **Decision:** top-1500 dense (ideal-candidate query) ∪ top-1500 lexical (literal JD query).
- **Why:** scoring is top-heavy (NDCG@10/50 = 80%), so the ranker must not be starved of true positives. Dense and sparse retrieval fail differently; the union has higher recall than either.
- **Alternatives:** dense only (misses exact-term matches); BM25 only (misses plain-language profiles — a stated trap); reciprocal-rank fusion (would give a single ordering, but the learned ranker already fuses via features).
- **Trade-off:** more candidates to embed online; recall is still capped by the narrow query text.

## D3. Two JD embeddings

- **Decision:** embed the literal JD and the "ideal candidate" paragraph; use the latter for dense retrieval, both as ranker features.
- **Why:** profile text is narrative, the JD is a bullet list. The ideal-candidate paragraph is closer in register to profiles, and the JD itself says the gap between "what it says and what it means" matters.
- **Trade-off:** extraction depends on string matching of paragraph markers ("imagining", "disqualifiers") in the docx — brittle if the JD text is edited.

## D4. Exact index and a small encoder (MiniLM)

- **Decision:** `IndexFlatIP` + `all-MiniLM-L6-v2` (384-d, CPU-friendly).
- **Why:** exact search over 100K×384 is milliseconds; a small encoder lets ~3K narratives be embedded online on CPU within budget.
- **Alternatives:** BGE/E5 (better quality, slower, bigger); HNSW/IVF (unnecessary at this scale); cross-encoder re-rank (expensive on CPU, and no labelled data to fine-tune).
- **Trade-off:** MiniLM truncates long inputs (≈256 word-pieces by default), so long narratives are represented by their beginning.

## D5. Honeypots: OR-logic hard filter, self-contained rules only

- **Decision:** five independent rules (1, 2, 4, 5, 6); any one fires → removed before the top-100 cut. Every rule uses only fields inside the candidate's own record.
- **Why:** ground truth forces honeypots to relevance 0 and > 10% in the top 100 is a disqualification — an asymmetric cost, so false positives are cheaper than false negatives. Averaging would dilute a single decisive red flag.
- **Alternatives:** weighted risk score + threshold; penalty inside the ranker; anomaly detector (isolation forest) over numeric features; no filter at all (the spec says a good ranker avoids honeypots naturally).
- **Trade-off:** the filter is broad (5.5% of the pool flagged vs ≈ 80 expected; rule 6 dominates) and two rules never fire on this data. Real honeypot recall is unknown.

## D6. (Reversed) Company-existence check inferred from the pool

- **Original decision:** flag a role that starts > 6 months before the company's *inferred* founding, taken as the ≈ 10th-percentile start date across ≥ 5 mentions (`company_stats.py`; Tier A/B split).
- **What went wrong:** the schema has no founding date, so the statistic measured "how early did this candidate start relative to peers". It flagged 17,477 candidates (17.5%), equally for AI/ML titles and everyone else, and ejected 7 of the eventual top 10.
- **Outcome:** rule removed on 2026-09-20; `company_stats.py` retained for reference only.
- **Lesson:** a heuristic that substitutes pool statistics for missing ground truth must be validated for *base rate* and *selectivity* before it is allowed to hard-filter. **Trade-off accepted:** the spec's "8 years at a company founded 3 years ago" pattern is no longer detected.

## D7. JD disqualifiers as soft features, not filters

- **Decision:** encode each "we do NOT want" clause as a 0/1 feature and let the ranker weight it.
- **Why:** detection is keyword-based over free text and therefore noisy; a wrong hard filter silently deletes a strong candidate, which hurts NDCG@10 more than a slightly wrong soft weight.
- **Trade-off:** the "hard" JD clause (pure research) has no stronger treatment than the soft ones unless the data teaches it — and in practice it did not: the fitted Ridge gives `is_pure_research`, `is_recent_wrapper_only` and `is_consulting_only` coefficients of ≈ 0, so those three disqualifiers are effectively not enforced.

## D8. Ridge over LightGBM LambdaRank

- **Decision:** `train_ranker.py` runs a 5-fold CV bake-off on the composite metric and saves the winner. Ridge won.
- **Why:** 150 labels, 16 features: a regularised linear model has far fewer degrees of freedom than boosted trees. Interpretable coefficients also help debug mis-rankings.
- **Alternatives:** logistic regression on relevance ≥ 3; pairwise models; hand-tuned weights.
- **Trade-off:** cannot learn interactions (e.g. "long notice matters only for otherwise-average candidates"); unscaled inputs make the shrinkage uneven (see known issues).

## D9. Synthetic, stratified eval set

- **Decision:** sample five archetypes (honeypot / stuffer / strong / mild / background) and assign relevance by archetype.
- **Why:** gets a supervised signal in minutes, covers every trap type explicitly, reproducible via seeds.
- **Trade-off:** the labels encode the author's definition of "fit"; the ranker learns that definition, not the hidden grader's. Treat CV scores as sanity checks only. The honeypot negatives were also sampled using flags that included the since-removed rule 3.

## D10. Reasoning: facts from the record, audited; LLM only as optional, gated polish

- **Original decision (v1):** a rule-based planner decided persona/trade-offs and a local Qwen 0.5B verbalised an *abstract* plan.
- **Problem found in audit:** the prompt held no concrete profile facts and nothing checked claims against the record, so output was generic ("strong alignment across semantic, technical, and readiness dimensions…") and sometimes wrong-sounding. It also doubled runtime (3 min 47 s, 7 GB).
- **Decision (now):** default `grounded` mode builds each explanation from the candidate's own record (title, employer, years, up to 3 listed JD skills with proficiency/months, a verbatim role-description quote, real concerns), tone by rank tier, and **audits every claim**; failure falls back to a minimal true sentence.
- **LLM polish is offline + cached + re-gated:** a small local model rewords drafts once; a strict gate accepts a rewrite only if every fact survives and nothing is added; ranking reads the cache and re-gates each entry, so no model loads in the graded step.
- **Alternatives:** free-form LLM over the full profile (hallucination, latency); pure templates (repetitive, but safe — this is the fallback); slot-filling where the LLM writes only connectives (implemented as `slotted-cached`, not yet populated).
- **Trade-off:** safe and fast but template-based — phrasing repeats across rows; the LLM rewrite covers only ≈ 38–39% of rows because small models drop the verbatim quote even when told not to.

## D11. Keep the legacy path selectable

- **Decision:** `--reasoning-mode llm` still runs the old planner + Qwen generation (lazy-imports torch), so the two approaches can be compared and the old behaviour reproduced.
- **Trade-off:** more code to maintain; its output quality is the reason it is not the default.

## D12. Stream and slim everything

- Regex-only ID scan of the candidates file (no JSON parse) in `rank.py`.
- Streaming `ParquetWriter` in 5K-row batches in `parse_candidates.py`.
- Nested objects as JSON strings in parquet.
- **Why:** 465 MB input and a 16 GB ceiling; keeps peak memory low and wall-clock short.

## D13. Deterministic by construction

Fixed seeds, explicit tie-break, hash-seeded template choice (`md5(candidate_id)`), pinned `AS_OF_DATE` (2026-06-30) for recency, greedy decoding for offline LLM rewrites, cache keyed by exact draft text. Same inputs ⇒ same CSV (modulo library/version drift).

## D14. Git LFS + force-push mirror to a Hugging Face Space

Deliverable includes a runnable sandbox. LFS keeps heavy parquet out of normal git objects; the workflow mirrors `main` to the Space on every push. Simple, but there is no test gate.

## D15. Availability as a post-model multiplier, and neutralising a wrong-signed coefficient

- **Context:** the JD says unreachable candidates ("hasn't logged in for 6 months, 5% response rate") must be down-weighted, but `behavioral_score` is one small feature inside a Ridge; its effect was ≈ 0.24 points on a 4–8 scale. Separately, the fitted Ridge rewarded *longer* notice periods (+0.16).
- **Decision:** after prediction, multiply by an availability factor (recency × response rate × notice; range 0.29–1.0) that can only lower a candidate's standing, and zero the notice feature at prediction time instead of retraining.
- **Why:** transparent, directly traceable to JD wording, leaves the trained pickle untouched (no retraining on rule-generated labels).
- **Alternatives:** retrain with monotonic constraints; engineer an explicit "unavailable" label; hand-edit coefficients.
- **Trade-off:** parameters are judgment values, not fitted; with scores tightly packed at ranks 50–100 a small multiplier change reorders the middle of the list.

## D16. Small-sample mode in `rank.py`

- **Decision:** with ≤ 1,500 candidate IDs, skip retrieval and score everyone; set `bm25_score = 0`.
- **Why:** the sandbox (Hugging Face Space) cannot hold the git-ignored 340 MB of indexes, and a retrieval step is pointless when the whole file fits in the shortlist.
- **Trade-off:** scores differ slightly from a full run (BM25 weight ≈ 0.004); and only IDs present in the precomputed `features.parquet` can be ranked at all.
