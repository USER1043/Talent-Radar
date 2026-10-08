# App & Deployment

## Streamlit app (`app/app.py`)

Launch: `streamlit run app/app.py` (from the repo root — relative paths like `submission.csv` and `artifacts/…` assume it).
The app adds the repo root to `sys.path` so it can import `features_utils`.

### Tab 1 — ⚙️ Pipeline Runner

1. User uploads a `.jsonl` / `.gz` file.
2. On **Run Ranking Pipeline** the upload is written to a temp dir and `rank.py` is invoked as a subprocess:
   `python rank.py --candidates <tmp>/candidates.jsonl[.gz] --out <tmp>/ranked_output.csv`.
3. On success: shows the CSV in a dataframe, offers a download button, and saves a copy to `./temp_preview_ranked.csv`
   so the Explorer tab can display it. On failure: shows `stderr`.

The five "Stage n/5" status messages and progress-bar values are **cosmetic** — they are set around a single blocking
`subprocess.run`, not driven by real pipeline progress.

Requirements to work: all `rank.py` artifacts (`faiss.index`, `bm25.pkl`, parquet files, `ranker_model.pkl`) present,
the uploaded IDs present in `features.parquet`, and the Qwen / MiniLM weights available.

### Tab 2 — 🔍 Candidate Explorer

- `load_data(csv_path)` (`@st.cache_data`) joins the submission with `features.parquet` and `honeypot_flags.parquet`, then re-runs
  `extract_candidate_features` per row so the UI can show trap flags and behaviour scores (overriding `is_consulting_only` and `behavioral_score` from the fresh computation).
- **Sidebar filters**: free-text search (ID / title / company) and a minimum-YOE slider.
- **Left column**: ranked table (Rank, ID, Title, Company, Final Score).
- **Right column**: card for a selected candidate with the reasoning, then two sub-tabs —
  *Performance & Availability* (YOE, notice vs the 30-day JD limit, behavioural score bar) and
  *Exclusions & Rules* (consulting-only, pure research, wrapper-only, non-coding-role flags).
- **Bottom**: scatter of final score vs behavioural score; YOE histogram.
- Four KPI cards on top. "Shortlist Size" (2,956) and "Valid Post-Honeypot" (2,556) are **hard-coded** numbers from one run, and "Total Pool"
  shows the number of rows in the displayed CSV (100), not the 100K pool.

If `temp_preview_ranked.csv` exists, the Explorer reads it instead of `submission.csv`; a button deletes it to revert.

### Testing the app

`pytest tests/test_streamlit_app.py` calls `load_data.__wrapped__()` (the un-cached function) and asserts the required columns exist.

## Hugging Face Space

The YAML front-matter at the top of `README.md` configures the Space:

```yaml
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.13"
app_file: app/app.py
```

`requirements.txt` pins `streamlit==1.45.1` (and `pandas==2.2.3`) to match — several commits in history are
dependency-alignment fixes for exactly this. Large artifacts are provided through Git LFS (`.gitattributes`).

## CI: `.github/workflows/sync_to_hf.yml`

On every push to `main`:

1. `actions/checkout@v4` with `fetch-depth: 0` and `lfs: true`.
2. Add a remote `hf` pointing at the Space repo, authenticated with the `HF_TOKEN` repository secret.
3. `git push hf main --force`.

Implications: the Space always mirrors GitHub `main` exactly (force-push overwrites any Space-side edits), the workflow does **not** run tests, and
a missing/expired `HF_TOKEN` fails the job. There is no separate CI job for `pytest`.

## Submission metadata (`submission_metadata.yaml`)

Mirrors the portal form: team, contact, repo and sandbox links, reproduce command, compute environment
(16 cores, 14 GB RAM, no GPU, no network during ranking, ~30–40 min precompute), AI-tools declaration, a methodology summary, and declarations.
Keep it in sync with behaviour changes (e.g. which model produces reasoning).

## Operational checklist

- Never commit `candidates.jsonl(.gz)`, `faiss.index`, `bm25.pkl` (they are in `.gitignore`).
- New large binary under `artifacts/`? Add it to `.gitattributes` (LFS) or `.gitignore` — pick deliberately.
- Pickles (`bm25.pkl`, `ranker_model.pkl`) execute code on load; only load artifacts you built or trust.
- Changing the embedding model requires rebuilding `faiss.index` and `jd_embeddings.npy`, **retraining** the ranker (similarity features shift), and re-tuning the 0.52–0.58 thresholds in `reasoning.py`.
