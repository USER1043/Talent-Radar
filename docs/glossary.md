# Glossary

| Term | Meaning in this project |
|---|---|
| **Active candidate** | A candidate ID present in the file passed to `rank.py --candidates` |
| **Behavioural score** | 0–1 composite of recency, responsiveness, speed, reliability, offer acceptance, availability, trust and GitHub signals (`features_utils.compute_behavioral_score`) |
| **BM25** | Lexical ranking function (`BM25Okapi`) used as the sparse retrieval leg and as a ranker feature |
| **Composite score** | `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` — the hackathon's grading formula |
| **Consulting-only** | Every career entry is at TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant or Capgemini |
| **Corroborated company (Tier A)** | Company named in ≥ 5 career entries across the pool; its inferred earliest start date is trusted |
| **Eval set** | 150 stratified candidates with rule-assigned relevance (0/1/3/5) used to train and select the ranker |
| **Honeypot** | Deliberately impossible profile planted in the pool; ground truth relevance 0; must be < 10% of the top 100 |
| **Ideal-candidate paragraph** | The JD section ("How to read between the lines") describing the imagined perfect hire; embedded separately |
| **Keyword stuffer** | Profile whose skill list looks perfect but whose role/career does not fit (e.g. HR Manager with AI skills) |
| **LFS** | Git Large File Storage — hosts `features.parquet` and `honeypot_flags.parquet` |
| **Narrative text** | headline + summary + career descriptions + skill names, with first-person phrasing rewritten |
| **Persona** | Reason-planner label (Elite Match, High Risk, …) that selects wording for a candidate's explanation |
| **Plain-language Tier 5** | Genuinely strong candidate who describes work without buzzwords |
| **Reason plan** | `ReasonPlan` dataclass: persona, decision analysis, evidence, neighbour differentiators |
| **Redrob signals** | 23 platform behaviour/availability fields in `redrob_signals` |
| **Sentinel (−1)** | "No data" marker in `offer_acceptance_rate` / `github_activity_score`; mapped to a neutral 0.5 |
| **Shortlist** | Union of FAISS top-1500 and BM25 top-1500 (~3K after overlap) |
| **Soft vs hard filter** | Soft = feature the ranker can weight (JD disqualifiers); hard = removal before ranking (honeypots) |
| **Title-chaser** | Average tenure < 18 months across ≥ 2 employers |
| **Wrapper-only** | LangChain/OpenAI-style skills with no pre-LLM ML background and < 1.5 years of experience |
