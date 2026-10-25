# FAQ — Deeper Questions About the Design

Short, direct answers to the questions contributors ask most. Each points to the doc with the full story.

## Big picture

**Q: In one paragraph, what does the system do?**
It narrows 100K profiles to a ~3K shortlist with a union of dense (FAISS/MiniLM) and lexical (BM25) retrieval, removes impossible "honeypot" profiles with OR-logic rules, scores the rest with a Ridge model over 16 features (semantic similarity, BM25, JD-disqualifier flags, behavioural score, experience), multiplies by an availability factor (inactivity, low response rate, long notice), sorts, keeps the top 100, and attaches a fact-audited 1–2 sentence explanation to each. Heavy work is precomputed so the graded run fits 5 CPU-minutes. → [architecture](architecture.md)

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

**Q: Do you check whether a company existed during someone's tenure?**
Not any more. The original rule inferred a founding date from other candidates' start dates, but with no real founding data it flagged 17.5% of the pool regardless of role and removed 7 of the eventual top 10, so it was dropped. Every remaining rule uses only fields inside the candidate's own record.

**Q: What are the false-positive risks?**
Noisy `duration_months` data, genuine concurrent part-time roles, and non-standard degree names. Mitigated by tolerances (2 months) but not eliminated; inspect `honeypot_reasons`.

## Reasoning

**Q: How do you stop reasoning from hallucinating?**
By construction. The default generator is deterministic: it fills sentence frames only with facts read from the candidate's own record (title, employer, years, listed skills with proficiency/months, a verbatim role-description quote, real concerns), then `audit_reasoning` re-checks every claim; a failure falls back to a minimal true sentence. No LLM is involved at ranking time. → [reasoning-engine](reasoning-engine.md)

**Q: Then where does the LLM come in?**
Optionally, offline. A small local model (Qwen2.5-1.5B/0.5B) rewords drafts; a strict gate accepts a rewrite only if it keeps the title/company relation, ≥ 2 skills exactly, every quote, every concern/number and the rank-tier verdict, and adds no numbers, names, praise or job-posting wording. Accepted rewrites are cached by draft text; at ranking time the cache is read and each entry is re-gated, so no model is loaded. About 38–39 of 100 rows get a rewrite; the rest use the grounded text.

**Q: Why not let the LLM write the text directly from the profile?**
It would be fluent but unverifiable, slower (the legacy LLM mode takes 3 min 47 s vs 1 min 47 s and 7 GB vs 4 GB), and a 0.5B model drifts: swapping who works where, adding praise words, dropping numbers. An earlier version fed only abstract labels to Qwen and produced generic text; that is why the default changed. → [design D10](design-decisions.md)

**Q: Why do adjacent candidates get different explanations?**
Different candidates surface different facts (skills, quotes, concerns), 5 × 4 sentence frames and 3 verdict phrasings per tier are chosen by `md5(candidate_id)`, and the LLM-rewritten rows add further variety. Phrasing still repeats in places (e.g. "the JD's 5-9 year band") — this is a known limitation of a template-based default.

**Q: What does the "slotted" mode do?**
The LLM writes only connecting words around placeholders like `[WHO]`/`[SKILLS]`; code fills the facts, so the model physically cannot alter them. The gate and cache-refill logic exist and are tested, but the committed cache holds no templates and there is no CLI entry to build them, so `slotted-cached` currently falls back to the grounded text.

## Engineering

**Q: Why store career/skills as JSON strings in parquet?**
Stable schema across batches, no sparse column explosion, and downstream readers only parse what they need.

**Q: Why scan IDs with a regex instead of parsing JSON?**
The input is 465 MB; the online run only needs to know *which* candidates are present, and a regex over each line is far cheaper than `json.loads`.

**Q: How is determinism ensured?**
Seeds (42/123), numeric tie-breaks, md5-based template selection, pinned `AS_OF_DATE` (2026-06-30) for recency, greedy decoding for offline rewrites, and a rewrite cache keyed by the exact draft text.

**Q: What breaks if I swap the embedding model?**
Everything that depends on cosine scale: `faiss.index`, `jd_embeddings.npy` and the ranker (trained on the old similarities). Rebuild the indexes, then retrain. (Note a fresh JD re-embedding already differs from the committed one by up to 0.095, so check ordering before and after.)

**Q: How would this scale to 10M candidates?**
Replace `IndexFlatIP` with an IVF-PQ or HNSW index, shard BM25 (or move to OpenSearch/Elasticsearch), keep the ranker linear and stateless, cache per-candidate features, and batch embedding online only for the shortlist. The offline/online split stays.

**Q: How would you evaluate this in production?**
Offline: labelled judgments with NDCG/MAP; online: A/B tests on recruiter engagement, plus offline-to-online correlation checks. The JD itself describes this evaluation loop as a first-90-day priority.

**Q: What's the first thing you'd fix?**
Measurement: the labels are rule-generated and the honeypot negatives came from a flag set that included a since-removed rule, so ranking quality against the hidden tiers is unmeasured. Next: the three JD disqualifiers with ≈ 0 coefficient, the FAISS/BM25 row-position alignment in `rank.py` for mid-sized subset inputs, and feature scaling. → [known-issues](known-issues-and-roadmap.md)

## Changes over time

**Q: What did the later audit change, and why?**
(1) Removed the company-founding honeypot rule — it flagged 17.5% of the pool irrespective of role and ejected 7 of the top 10. (2) Added the availability multiplier and neutralised the Ridge's wrong-signed notice coefficient. (3) Replaced the abstract-label Qwen reasoning with a fact-grounded, audited generator, plus an offline gated LLM-rewrite cache. (4) Added a small-sample mode so the sandbox runs without the big indexes. (5) Added tests for the reasoning stack. → [design-decisions](design-decisions.md)

**Q: How would you check that the honeypot filter is doing its job?**
Per-rule flag counts and rates (the table in [honeypot-detection](honeypot-detection.md)), the flag rate split by AI/ML vs other titles (a good rule should not flag both equally), how many top-ranked candidates a rule would eject, and a manual read of `honeypot_reasons` for a sample. Rule 6 flags 5,248 of 5,542, so it deserves the first look.
