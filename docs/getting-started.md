# Getting Started

## Prerequisites

| Need | Detail |
|---|---|
| Python | 3.13 (the HF Space pins `python_version: "3.13"`) |
| Hardware | CPU only. The graded run is bounded to ≤ 16 GB RAM and ≤ 5 minutes wall-clock |
| Disk | A few GB: `faiss.index` ≈ 150 MB, `bm25.pkl` ≈ 185 MB, `features.parquet` ≈ 87 MB (LFS) |
| Network | Needed once to download `all-MiniLM-L6-v2` and `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face; the ranking run itself is intended to be offline once weights are cached |
| Git LFS | `artifacts/features.parquet` and `artifacts/honeypot_flags.parquet` are LFS objects. Without `git lfs pull` you get ~130-byte pointer files and everything downstream fails |
| Data | `candidates.jsonl` or `candidates.jsonl.gz` (~465 MB). **Not in the repo.** Place it in the repo root |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git lfs install && git lfs pull          # materialise the parquet artifacts
```

Dependencies are pinned in `requirements.txt` and grouped by purpose:
pandas/pyarrow/scikit-learn (data + Ridge), sentence-transformers/faiss-cpu/rank-bm25
(retrieval), lightgbm (candidate ranker for the bake-off), transformers/torch/accelerate
(Qwen reasoning), streamlit (UI), python-docx (reading the JD).

## What can I run from a fresh clone?

| Goal | Works from a clone? | Why |
|---|---|---|
| Read/explore `submission.csv` in the dashboard | ✅ after `git lfs pull` | Needs `features.parquet`, `honeypot_flags.parquet`, `submission.csv` only |
| `pytest tests/test_streamlit_app.py` | ✅ after `git lfs pull` | Same data as above |
| `pytest tests/test_no_stuffing_regression.py` | ⚠️ partially | Two fixtures need `sample_submission.csv` from the hackathon bundle, which is not committed |
| `python validate_local.py --csv submission.csv` | ✅ | Pure CSV checks; add `--candidates` for ID-existence check |
| `python rank.py ...` | ❌ until you rebuild indexes | Needs `faiss.index` + `bm25.pkl` (gitignored) and the candidates file |
| Full pipeline from scratch | ✅ with the candidates file | Run the six offline steps below |

## Rebuild everything (offline, no time limit, ~30–40 min on 16 cores)

Run from the repo root. Order matters — each step consumes the previous step's artifacts.

```bash
# 1. Flatten 100K JSONL records → artifacts/features.parquet
python parse_candidates.py --candidates ./candidates.jsonl.gz --out artifacts/features.parquet

# 2. Per-company earliest-plausible-start statistics
python company_stats.py --features artifacts/features.parquet --out artifacts/company_stats.parquet

# 3. Honeypot flags (needs 1 and 2)
python honeypot_flags.py --features artifacts/features.parquet \
    --company-stats artifacts/company_stats.parquet --out artifacts/honeypot_flags.parquet

# 4. Embeddings + FAISS + BM25 + JD embeddings (needs 1 and job_description.docx)
python build_index.py            # → artifacts/faiss.index, bm25.pkl, jd_embeddings.npy

# 5. Stratified evaluation set (needs 1, 3, 4)
python label_eval_set.py         # → artifacts/eval_set.csv

# 6. Train/select the ranker (needs 1, 4, 5)
python train_ranker.py           # → artifacts/ranker_model.pkl
```

## Produce a ranking

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_local.py --csv submission.csv --candidates ./candidates.jsonl
python validate_submission.py submission.csv       # official format validator
pytest -q
```

`rank.py` prints one line per stage and the total wall-clock time. Optional flags
override every artifact path (`--features`, `--honeypots`, `--faiss-index`, `--bm25`,
`--jd-embeds`, `--job-description`, `--ranker-model`).

## Run the dashboard

```bash
streamlit run app/app.py
```

Two tabs: **Pipeline Runner** (upload a candidates file → runs `rank.py` in a
subprocess) and **Candidate Explorer** (browse `submission.csv`, inspect reasons,
trap checks, charts). Details in [app-and-deployment.md](app-and-deployment.md).

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `ArrowInvalid` / "Parquet magic bytes not found" | LFS pointer files | `git lfs pull` |
| `FileNotFoundError: artifacts/faiss.index` | Index is gitignored | Run step 4 |
| `ModuleNotFoundError: docx` | `python-docx` missing | `pip install -r requirements.txt` (`extract_jd_texts` otherwise silently falls back to a 2-sentence JD — see known issues) |
| First `rank.py` run is slow / hangs offline | Qwen + MiniLM weights downloading | Pre-download once with network, then run offline |
| `LLM Batch generation failed: …` in logs | Qwen could not load or generate | Expected degradation path: deterministic templates are used instead |
| Pipeline Runner finds only a few candidates | Uploaded IDs not in `features.parquet` | Rows are filtered to IDs present in the precomputed table; see known issues for index-alignment caveat |
