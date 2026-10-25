# Glossary

| Term | Meaning in this project |
|---|---|
| **Audit** | `reasoning_grounded.audit_reasoning`: re-checks each claim in a reasoning string (skills, quotes, years, response rate, notice, title/company) against the candidate's record |
| **Active candidate** | A candidate ID present in the file passed to `rank.py --candidates` |
| **Availability multiplier** | Post-model factor (0.29–1.0) from last-active recency, recruiter response rate and notice period; can only lower a score |
| **Behavioural score** | 0–1 composite of recency, responsiveness, speed, reliability, offer acceptance, availability, trust and GitHub signals (`features_utils.compute_behavioral_score`) |
| **BM25** | Lexical ranking function (`BM25Okapi`) used as the sparse retrieval leg and as a ranker feature |
| **Composite score** | `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` — the hackathon's grading formula |
| **Consulting-only** | Every career entry is at TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant or Capgemini |
| **Corroborated company (Tier A)** | *Retired concept* — company named in ≥ 5 career entries; its inferred earliest start date used to be trusted by honeypot rule 3 (removed) |
| **Draft** | The deterministic text from `compose_reasoning`; the base that LLM rewrites must preserve, and the key of the rewrite cache |
| **Eval set** | 150 stratified candidates with rule-assigned relevance (0/1/3/5) used to train and select the ranker |
| **Gate** | `gate_rewrite` / `gate_slotted`: strict checks a rewritten reasoning must pass before it replaces the draft |
| **Grounded** | Built only from facts in the candidate's own record and verified by the audit |
| **Honeypot** | Deliberately impossible profile planted in the pool; ground truth relevance 0; must be < 10% of the top 100 |
| **Ideal-candidate paragraph** | The JD section ("How to read between the lines") describing the imagined perfect hire; embedded separately |
| **Keyword stuffer** | Profile whose skill list looks perfect but whose role/career does not fit (e.g. HR Manager with AI skills) |
| **LFS** | Git Large File Storage — hosts `features.parquet` and `honeypot_flags.parquet` |
| **JD-embeddings gotcha** | Re-embedding the JD yields values that differ from the committed `jd_embeddings.npy` (max abs diff 0.095); don't overwrite the committed file |
| **Narrative text** | headline + summary + career descriptions + skill names, with first-person phrasing rewritten |
| **Persona** | Label from the *legacy* reason planner (Elite Match, High Risk, …) used by `--reasoning-mode llm` |
| **Plain-language Tier 5** | Genuinely strong candidate who describes work without buzzwords |
| **Reason plan** | Legacy `ReasonPlan` dataclass: persona, decision analysis, evidence, neighbour differentiators |
| **Redrob signals** | 23 platform behaviour/availability fields in `redrob_signals` |
| **Sentinel (−1)** | "No data" marker in `offer_acceptance_rate` / `github_activity_score`; mapped to a neutral 0.5 |
| **Shortlist** | Union of FAISS top-1500 and BM25 top-1500 (2,969 on the full pool) |
| **Slot** | Placeholder such as `[WHO]` or `[SKILLS]` in a slotted template; filled by code from the record, never by the LLM |
| **Small-sample mode** | `rank.py` with ≤ 1,500 candidate IDs: retrieval skipped, every candidate scored, BM25 feature = 0 |
| **Soft vs hard filter** | Soft = feature the ranker can weight (JD disqualifiers); hard = removal before ranking (honeypots) |
| **Title-chaser** | Average tenure < 18 months across ≥ 2 employers |
| **Wrapper-only** | LangChain/OpenAI-style skills with no pre-LLM ML background and < 1.5 years of experience |
