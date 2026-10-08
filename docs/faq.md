# FAQ — Deeper Questions About the Design

Short, direct answers to the questions contributors ask most. Each points to the doc with the full story.

## Big picture

**Q: In one paragraph, what does the system do?**
It narrows 100K profiles to a ~3K shortlist with a union of dense (FAISS/MiniLM) and lexical (BM25) retrieval, removes impossible "honeypot" profiles with OR-logic rules, scores the rest with a Ridge model over 16 features (semantic similarity, BM25, JD-disqualifier flags, behavioural score, notice penalty, experience), sorts, keeps the top 100, and attaches a grounded 1–2 sentence explanation to each. Heavy work is precomputed so the graded run fits 5 CPU-minutes. → [architecture](architecture.md)

**Q: Why not just rank by embedding similarity to the JD?**
The dataset is built to punish that: keyword stuffers have perfect skill lists but the wrong role, plain-language strong candidates never use the buzzwords, and honeypots are impossible profiles that can look great. Similarity is one feature among sixteen; title coherence, disqualifier flags, and availability signals correct for it. → [retrieval-and-ranking](retrieval-and-ranking.md)

**Q: Why is the scoring formula so influential on the design?**
`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` puts 80% of the weight on the first 10/50 positions. So the pipeline maximises recall into the shortlist, then spends its effort on careful ordering, and uses hard filters only where a mistake is catastrophic (honeypots).

## Retrieval

**Q: Why two retrievers?**
Dense retrieval generalises across wording but can miss rare exact terms; BM25 does the opposite. Their errors are mostly independent, so the union raises recall. The ranker later decides how much to trust each, since both scores are features.

**Q: Why query FAISS with the "ideal candidate" paragraph and BM25 with the literal JD?**
Dense similarity benefits from text that reads like a profile (the narrative paragraph), while BM25 benefits from the vocabulary-rich bullet list. Both embeddings are still used as ranker features.

**Q: What does L2-normalising before `IndexFlatIP` do?**
For unit vectors, inner product equals cosine similarity, so the "inner product" index returns cosine ranks and scores in [−1, 1].

**Q: How does BM25 work, in a sentence?**
A TF–IDF-style score where term frequency saturates (`k1`) and document length is normalised (`b`), so a term repeated many times in a long document doesn't dominate. (`rank_bm25.BM25Okapi` with default parameters.)

**Q: Why exact search instead of HNSW/IVF?**
100K × 384 floats is small; exact search is milliseconds and has no recall loss to reason about.

**Q: What could still be missed?**
A strong candidate far from the JD in both embedding and lexical space. Mitigations: multi-query retrieval per JD requirement, or a wider K. → [known-issues §B6](known-issues-and-roadmap.md)

## Features & model

**Q: How is the behavioural score built, and why those weights?**
Eight signals (recency with 90-day half-life, response rate, response speed, interview completion, offer acceptance, open-to-work, verification trust, GitHub activity), weights 0.15/0.15/0.10/0.15/0.15/0.10/0.10/0.10, sentinel values mapped to neutral 0.5. Weights are hand-set priors; the JD says unreachable candidates should be down-weighted but doesn't say by how much.

**Q: Why Ridge instead of LightGBM LambdaRank?**
With 150 labelled rows, a regularised linear model generalises better and is interpretable; the script actually runs both in 5-fold CV and saves the winner (Ridge won). If you add many more labels, re-run the bake-off — LambdaRank may win. → [design D8](design-decisions.md)

**Q: Ridge is a regressor — isn't ranking a different problem?**
Yes, it's a pointwise approach: regress graded relevance, then sort by prediction. It's a strong baseline when labels are few; pairwise/listwise losses (LambdaRank) are the natural upgrade when you have enough queries and labels.

**Q: What are NDCG, MAP and P@10?**
NDCG: gain of each result discounted by `log2(position+1)`, normalised by the best possible ordering. MAP: mean over relevant items of precision at the position they appear. P@10: share of relevant results in the first ten.

**Q: Are the eval metrics trustworthy?**
They're a regression check, not a leaderboard estimate: labels come from sampling rules and overlap with some features. → [evaluation](evaluation-and-testing.md), [known-issues §B1](known-issues-and-roadmap.md)

**Q: Why aren't honeypot flags a ranker feature?**
They're removed by a hard filter anyway; as a feature they'd be redundant on candidates who survive the filter, and the eval set's honeypots would let the model learn to cheat the label definition.

## Honeypots

**Q: Why OR-logic rather than a score?**
One impossible fact (an expert skill with 0 months of use, overlapping jobs) is enough to disprove a profile. Averaging with normal-looking signals would hide it. → [honeypot-detection](honeypot-detection.md)

**Q: Why hard-filter instead of penalise?**
Because >10% honeypots in the top 100 is a disqualification and they are ground-truth zeros; a penalty can be outweighed by a high similarity score, a filter cannot.

**Q: How do you check a company's existence without external data?**
By inferring a lower bound from the pool: the ~10th-percentile start date across ≥ 5 mentions. Rare companies are skipped for that rule.

**Q: What are the false-positive risks?**
Noisy `duration_months` data, genuine concurrent part-time roles, and non-standard degree names. Mitigated by tolerances (2 months) but not eliminated; inspect `honeypot_reasons`.

## Reasoning

**Q: How do you stop the LLM from hallucinating?**
It never sees the raw profile — only a closed set of facts produced by deterministic rules (rank, persona, constraint, differentiator), plus few-shot examples, greedy decoding, and post-validation. If the model fails entirely, rule-based templates take over. → [reasoning-engine](reasoning-engine.md)

**Q: Why do adjacent candidates get different explanations?**
`plan_reason` compares each candidate to the ones ranked just above and below and records the specific reason they beat/lose (notice period, semantic match, experience, coding). Templates are also chosen by `md5(candidate_id)`.

**Q: Why such a small model?**
Network is off and the whole run, including 100 generations, must fit in 5 CPU-minutes. A 0.5B instruct model plus structured facts is a pragmatic point on that curve.

## Engineering

**Q: Why store career/skills as JSON strings in parquet?**
Stable schema across batches, no sparse column explosion, and downstream readers only parse what they need.

**Q: Why scan IDs with a regex instead of parsing JSON?**
The input is 465 MB; the online run only needs to know *which* candidates are present, and a regex over each line is far cheaper than `json.loads`.

**Q: How is determinism ensured?**
Seeds (42/123), numeric tie-breaks, md5-based template selection, pinned "today" (2026-06-30) for recency, greedy decoding.

**Q: What breaks if I swap the embedding model?**
Everything that depends on cosine scale: `faiss.index`, `jd_embeddings.npy`, the ranker (trained on old similarities), and the 0.52–0.58 thresholds in `reasoning.py`. Rebuild and retrain in that order.

**Q: How would this scale to 10M candidates?**
Replace `IndexFlatIP` with an IVF-PQ or HNSW index, shard BM25 (or move to OpenSearch/Elasticsearch), keep the ranker linear and stateless, cache per-candidate features, and batch embedding online only for the shortlist. The offline/online split stays.

**Q: How would you evaluate this in production?**
Offline: labelled judgments with NDCG/MAP; online: A/B tests on recruiter engagement, plus offline-to-online correlation checks. The JD itself describes this evaluation loop as a first-90-day priority.

**Q: What's the first thing you'd fix?**
The FAISS/BM25 row-position alignment in `rank.py` for subset inputs (A1), then human-reviewed labels and feature scaling. → [known-issues](known-issues-and-roadmap.md)
