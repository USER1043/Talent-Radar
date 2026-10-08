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

## D5. Honeypots: OR-logic hard filter

- **Decision:** six independent rules; any one fires → removed from candidates before the top-100 cut.
- **Why:** ground truth forces honeypots to relevance 0 and > 10% in the top 100 is a disqualification — an asymmetric cost, so false positives are cheaper than false negatives. Averaging would dilute a single decisive red flag.
- **Alternatives:** weighted risk score + threshold; penalty inside the ranker; anomaly detector (isolation forest) over numeric features.
- **Trade-off:** some genuine candidates with noisy data (e.g. a 3-month `duration_months` mismatch) are lost.

## D6. Company-existence check inferred from the pool

- **Decision:** 10th-percentile start date per company, trusted only with ≥ 5 mentions.
- **Why:** no external founding-date source is available offline; the median would flag ~half of small groups by construction.
- **Trade-off:** rare companies get no timeline check (they rely on the other five rules); a fabricated company mentioned by many honeypots could "corroborate" itself.

## D7. JD disqualifiers as soft features, not filters

- **Decision:** encode each "we do NOT want" clause as a 0/1 feature and let the ranker weight it.
- **Why:** detection is keyword-based over free text and therefore noisy; a wrong hard filter silently deletes a strong candidate, which hurts NDCG@10 more than a slightly wrong soft weight.
- **Trade-off:** the "hard" JD clause (pure research) has no stronger treatment than the soft ones unless the data teaches it.

## D8. Ridge over LightGBM LambdaRank

- **Decision:** `train_ranker.py` runs a 5-fold CV bake-off on the composite metric and saves the winner. Ridge won.
- **Why:** 150 labels, 16 features: a regularised linear model has far fewer degrees of freedom than boosted trees. Interpretable coefficients also help debug mis-rankings.
- **Alternatives:** logistic regression on relevance ≥ 3; pairwise models; hand-tuned weights.
- **Trade-off:** cannot learn interactions (e.g. "long notice matters only for otherwise-average candidates"); unscaled inputs make the shrinkage uneven (see known issues).

## D9. Synthetic, stratified eval set

- **Decision:** sample five archetypes (honeypot / stuffer / strong / mild / background) and assign relevance by archetype.
- **Why:** gets a supervised signal in minutes, covers every trap type explicitly, reproducible via seeds.
- **Trade-off:** the labels encode the author's definition of "fit"; the ranker learns that definition, not the hidden grader's. Treat CV scores as sanity checks only.

## D10. Plan first, narrate second (reasoning)

- **Decision:** a rule-based planner decides persona, trade-offs, evidence and neighbour differentiators; an LLM (or template) only verbalises that plan.
- **Why:** keeps explanations consistent with the real rank and features; hard to hallucinate when the model never sees the raw profile; testable without a model.
- **Alternatives:** free-form LLM over the full profile (hallucination risk, latency); pure templates (repetitive, flagged as duplicates).
- **Trade-off:** explanations are only as specific as the plan's few fields — they rarely cite concrete projects or companies.

## D11. Local 0.5B model, one batched generation, fallback on any exception

- **Decision:** `Qwen2.5-0.5B-Instruct` on CPU, all 100 prompts in one `generate`, deterministic templates if anything fails.
- **Why:** no network/API allowed during ranking; time budget leaves ~2–3 minutes for 100 explanations; the pipeline must always produce a valid CSV.
- **Trade-off:** small-model fluency issues; a failure is silent apart from a log line, so output style can change between environments.

## D12. Stream and slim everything

- Regex-only ID scan of the candidates file (no JSON parse) in `rank.py`.
- Streaming `ParquetWriter` in 5K-row batches in `parse_candidates.py`.
- Nested objects as JSON strings in parquet.
- **Why:** 465 MB input and a 16 GB ceiling; keeps peak memory low and wall-clock short.

## D13. Deterministic by construction

Fixed seeds, explicit tie-break, hash-seeded template choice (`md5(candidate_id)`), pinned "today" for recency decay, greedy decoding. Same inputs ⇒ same CSV (modulo library/version drift).

## D14. Git LFS + force-push mirror to a Hugging Face Space

Deliverable includes a runnable sandbox. LFS keeps heavy parquet out of normal git objects; the workflow mirrors `main` to the Space on every push. Simple, but there is no test gate.
