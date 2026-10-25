# Getting Started

## Prerequisites

| Need | Detail |
|---|---|
| Python | 3.13 (the HF Space pins `python_version: "3.13"`) |
| Hardware | CPU only. Graded run is bounded to ≤ 16 GB RAM and ≤ 5 minutes wall-clock; measured 1 min 47 s / 4 GB on 16 cores |
| Disk | A few GB: `faiss.index` ≈ 153 MB, `bm25.pkl` ≈ 186 MB, `features.parquet` ≈ 87.5 MB (LFS) |
| Network | Needed once to download `all-MiniLM-L6-v2` (always) and, only for the LLM modes, `Qwen/Qwen2.5-*-Instruct` weights. After caching, set `HF_HUB_OFFLINE=1` |
| Git LFS | `artifacts/features.parquet` and `artifacts/honeypot_flags.parquet` are LFS objects. Without `git lfs pull` you get ~130-byte pointer files and everything downstream fails |
| Data | `candidates.jsonl` or `.jsonl.gz` (~465 MB). **Not in the repo** (git-ignored). Place it in the repo root |

## Install

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tested route: uv pip install -r requirements.txt --torch-backend cpu
git lfs install && git lfs pull          # materialise the parquet artifacts
# pre-cache the embedding model so ranking can run offline:
python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"
```

`requirements.txt` groups: pandas/pyarrow/scikit-learn (data + Ridge), sentence-transformers/faiss-cpu/rank-bm25 (retrieval), lightgbm (the other contender in
the model bake-off), transformers/torch/accelerate (only the LLM reasoning modes and offline rewrite build), streamlit (UI), python-docx (reading the JD).

## What can I run from a fresh clone?

| Goal | Works? | Why |
|---|---|---|
| Browse `submission.csv` in the dashboard | ✅ after `git lfs pull` | Needs `features.parquet`, `honeypot_flags.parquet`, `submission.csv` |
| `pytest tests/test_reasoning_*.py` | ✅ | Pure-Python tests with injected fake generators |
| `pytest tests/test_streamlit_app.py` | ✅ after `git lfs pull` | Uses the parquet artifacts |
| `pytest tests/test_no_stuffing_regression.py` | ⚠️ 1 setup error | Needs `sample_submission.csv` from the organisers (not committed) |
| `python validate_local.py --csv submission.csv` | ✅ | CSV checks; add `--candidates` for ID existence |
| `python rank.py` on ≤ 1,500 pool candidates | ✅ | Small-sample mode skips FAISS/BM25 (still needs MiniLM + parquet + ranker). IDs must be in `features.parquet` |
| `python rank.py` on the full pool | ❌ until you build the indexes | Needs `faiss.index` + `bm25.pkl` (git-ignored) and the 100K candidates file |

## Reproduce the submission

```bash
# 1. Put the pool next to rank.py (git-ignored)
cp /path/to/candidates.jsonl ./candidates.jsonl

# 2. One-off: search indexes (~37 min on 16 cores). NOTE the --out-jd-embeds override — see "JD embeddings" below.
python build_index.py --out-jd-embeds artifacts/jd_embeddings.rebuilt.npy

# 3. The graded step
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
#    same, reading precomputed LLM rewrites (no model needed):
#    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --reasoning-mode grounded-cached

# 4. Validate
python validate_submission.py submission.csv
python validate_local.py --csv submission.csv --candidates ./candidates.jsonl
pytest -q
```

### JD embeddings gotcha

`rank.py` reads the **committed** `artifacts/jd_embeddings.npy`, which is what the ranker was trained against. `build_index.py` writes that same path by default, but
re-embedding the JD yields *different* values (max abs difference 0.095; cause not determined; the re-embedding itself is deterministic), which shifts scores slightly.
Hence the `--out-jd-embeds` override above. Don't overwrite the committed file unless you also retrain the ranker.

## Regenerating other artifacts (optional)

| Step | Command | Time | Notes |
|---|---|---|---|
| Features | `python parse_candidates.py --candidates ./candidates.jsonl --out artifacts/features.parquet` | ~20 s | Reproduces the committed file exactly |
| Honeypot flags | `python honeypot_flags.py --features artifacts/features.parquet --out artifacts/honeypot_flags.parquet` | ~5 s | 5,542 flagged (5.5%). **No `--company-stats` any more** |
| Indexes | `python build_index.py …` | ~37 min | Writes `faiss.index`, `bm25.pkl` |
| Eval labels | `python label_eval_set.py` | not timed | **Overwrites `artifacts/eval_set.csv`**; needs `faiss.index` and flags |
| Ranker | `python train_ranker.py` | not timed | **Overwrites `artifacts/ranker_model.pkl`**; 5-fold CV Ridge vs LightGBM |
| LLM rewrite cache | `LLM_DTYPE=bfloat16 python rank.py … --reasoning-mode grounded-llm --llm-model Qwen/Qwen2.5-1.5B-Instruct --rewrite-cache artifacts/reasoning_rewrites.json` | ~17 min (1.5B) | Offline build; optional |

`company_stats.py` is **no longer part of the pipeline** (see [honeypot-detection](honeypot-detection.md)).

## Run the dashboard

```bash
streamlit run app/app.py
```

Two tabs: **Pipeline Runner** (upload a candidates file → runs `rank.py` in a subprocess) and **Candidate Explorer** (browse `submission.csv`, inspect reasoning,
trap checks, charts). Details in [app-and-deployment.md](app-and-deployment.md).

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `ArrowInvalid` / "Parquet magic bytes not found" | LFS pointer files | `git lfs pull` |
| `FileNotFoundError: artifacts/faiss.index` | Index is git-ignored; file has > 1,500 IDs | Run `build_index.py` (or use ≤ 1,500 IDs) |
| `ModuleNotFoundError: docx` | `python-docx` missing | `pip install -r requirements.txt` (`extract_jd_texts` otherwise silently falls back to a 2-sentence JD) |
| Hangs / fails offline loading MiniLM | Weights not in HF cache | Download once online; run with `HF_HUB_OFFLINE=1` |
| Empty `submission.csv` | None of the supplied IDs are in `features.parquet` | Candidate file must be drawn from the 100K pool |
| "LLM rewrite unavailable (…)" | `grounded-llm` could not load a model | Expected degradation: grounded drafts are used |
| "Cached LLM rewrites: 0/100 rows used" | Draft text or ranks changed, so cache keys miss | Rebuild the cache; fallback is the grounded text |
