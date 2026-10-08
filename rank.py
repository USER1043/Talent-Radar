"""
@file Graded entry point for retrieving, scoring, and ranking candidate profiles.
@package online_ranking
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from build_index import tokenize, extract_jd_texts
from features_utils import (
    apply_availability_multiplier,
    compute_availability_multiplier,
    extract_candidate_features,
)
from reasoning_grounded import generate_reasonings_grounded


# Opens a candidate jsonl or jsonl.gz file transparently.
def open_candidates(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# Main execution flow for online candidate ranking.
def main():
    import time
    pipeline_start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--features", default=Path("artifacts/features.parquet"), type=Path)
    ap.add_argument("--honeypots", default=Path("artifacts/honeypot_flags.parquet"), type=Path)
    ap.add_argument("--faiss-index", default=Path("artifacts/faiss.index"), type=Path)
    ap.add_argument("--bm25", default=Path("artifacts/bm25.pkl"), type=Path)
    ap.add_argument("--jd-embeds", default=Path("artifacts/jd_embeddings.npy"), type=Path)
    ap.add_argument("--job-description", default=Path("job_description.docx"), type=Path)
    ap.add_argument("--ranker-model", default=Path("artifacts/ranker_model.pkl"), type=Path)
    ap.add_argument(
        "--reasoning-mode", choices=["grounded", "grounded-cached", "grounded-llm", "slotted-cached", "llm"], default="grounded",
        help="grounded: deterministic, fact-checked text built from each candidate's own record (default); "
             "grounded-cached: precomputed LLM rewrites from --rewrite-cache, each re-checked by a strict fact gate, else the grounded text (no model needed); "
             "slotted-cached: cached LLM-written connecting words with placeholders (--rewrite-cache); the code re-checks each template and fills the facts from the record itself (no model needed); "
             "grounded-llm: OFFLINE build step, an LLM rewrites each grounded draft and accepted rewrites are saved to --rewrite-cache (slow); "
             "llm: legacy Qwen2.5-0.5B generation (needs the model cached locally, ~2 extra minutes).")
    ap.add_argument("--rewrite-cache", type=Path, default=Path("artifacts/reasoning_rewrites.json"),
                    help="grounded-cached reads this file; grounded-llm writes accepted rewrites into it.")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="grounded-llm only: model used for the offline rewrite build.")
    ap.add_argument("--reasoning-diagnostics", type=Path, default=None,
                    help="grounded-llm only: write per-row accept/fallback details to this JSON file.")
    args = ap.parse_args()

    print("Stage 1: Loading candidate IDs from candidates file...")
    active_ids = set()
    import re
    cid_re = re.compile(r'"candidate_id"\s*:\s*"(CAND_\d{7})"')
    with open_candidates(args.candidates) as f:
        for line in f:
            m = cid_re.search(line)
            if m:
                active_ids.add(m.group(1))
    print(f"Loaded {len(active_ids)} active candidate IDs.")

    print("Stage 2: Loading cached artifacts...")
    df_feat = pd.read_parquet(args.features)
    df_hp = pd.read_parquet(args.honeypots)
    
    # Filter features to only active candidate IDs
    df_feat = df_feat[df_feat["candidate_id"].isin(active_ids)].reset_index(drop=True)
    df_hp = df_hp[df_hp["candidate_id"].isin(active_ids)].reset_index(drop=True)

    print("Stage 3: Performing hybrid retrieval...")
    # Load JD embedding and FAISS index
    jd_embeds = np.load(args.jd_embeds)
    ideal_jd_emb = jd_embeds[1:2].astype("float32")
    faiss.normalize_L2(ideal_jd_emb)

    small_sample = len(active_ids) <= 1500
    bm25_all_scores = None
    if small_sample:
        # Sandbox / small-sample mode: score every candidate directly. The FAISS and BM25 indexes are built
        # over the full 100K pool (git-ignored, so absent e.g. on the Hugging Face Space) and are not needed
        # here. The BM25 feature is set to 0 in this mode (its ranker weight is only ~0.004).
        print(f"Small candidate file ({len(active_ids)} <= 1500): retrieval skipped, scoring every candidate.")
        shortlist_df = df_feat.copy().reset_index(drop=True)
    else:
        index = faiss.read_index(str(args.faiss_index))

        # Retrieve top 1500 semantic matches
        _, faiss_indices = index.search(ideal_jd_emb, 1500)
        faiss_set = set(faiss_indices[0])

        # Retrieve top 1500 lexical matches
        with open(args.bm25, "rb") as f:
            bm25 = pickle.load(f)

        literal_jd, _ = extract_jd_texts(args.job_description)
        jd_tokens = tokenize(literal_jd)
        bm25_all_scores = bm25.get_scores(jd_tokens)

        bm25_sorted_indices = np.argsort(bm25_all_scores)[::-1]
        bm25_set = set(bm25_sorted_indices[:1500])

        # Union of indices
        shortlist_indices = list(faiss_set.union(bm25_set))
        print(f"Retrieved shortlist size: {len(shortlist_indices)} unique candidates.")

        shortlist_df = df_feat[df_feat.index.isin(shortlist_indices)].copy()
        shortlist_df = shortlist_df[shortlist_df["candidate_id"].isin(active_ids)].reset_index(drop=True)

    print(f"Active shortlist size after filtering: {len(shortlist_df)} candidates.")
    
    if len(shortlist_df) == 0:
        print("Warning: No active candidates found in features dataset. Writing empty submission.")
        pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"]).to_csv(args.out, index=False)
        return

    print("Stage 4: Computing structured candidate features...")
    features_list = []
    for _, row in shortlist_df.iterrows():
        features_list.append(extract_candidate_features(row))
    feats_df = pd.DataFrame(features_list)

    print("Stage 5: Embedding shortlist candidates and computing similarities...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    shortlist_narratives = shortlist_df["narrative_text"].tolist()
    shortlist_embs = model.encode(shortlist_narratives, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(shortlist_embs)

    literal_jd_emb = jd_embeds[0:1].astype("float32")
    faiss.normalize_L2(literal_jd_emb)

    feats_df["semantic_sim_to_jd"] = np.dot(shortlist_embs, literal_jd_emb.T).flatten()
    feats_df["semantic_sim_to_ideal"] = np.dot(shortlist_embs, ideal_jd_emb.T).flatten()

    # Map BM25 scores
    # Maps candidate ID to its index in original df_feat
    if bm25_all_scores is None:  # small-sample mode: BM25 retrieval skipped
        feats_df["bm25_score"] = 0.0
    else:
        feat_id_to_idx = {cid: idx for idx, cid in enumerate(df_feat["candidate_id"])}
        feats_df["bm25_score"] = [bm25_all_scores[feat_id_to_idx[cid]] for cid in shortlist_df["candidate_id"]]

    # Assign features back to shortlist_df for use downstream in reasoning.py
    for col in feats_df.columns:
        shortlist_df[col] = feats_df[col].values

    print("Stage 6: Scoring candidates with trained ranker...")
    with open(args.ranker_model, "rb") as f:
        model_payload = pickle.load(f)
    ranker = model_payload["model"]
    feature_names = model_payload["feature_names"]

    X_df = feats_df[feature_names].copy()
    # The Ridge fit a POSITIVE coefficient (+0.16) on notice_period_penalty, i.e. it
    # rewarded longer notice periods, which contradicts the JD. Neutralise that term
    # here; the correct (penalising) direction is applied in the availability
    # multiplier below. The trained model file is left untouched.
    if "notice_period_penalty" in X_df.columns:
        X_df["notice_period_penalty"] = 0.0
    X = X_df.to_numpy()
    shortlist_df["model_score"] = ranker.predict(X)

    # Down-weight candidates who are effectively unavailable (long inactive and/or
    # rarely reply to recruiters), as the JD asks. Applied after the ranker so the
    # trained model is unchanged.
    shortlist_df["availability_multiplier"] = shortlist_df.apply(compute_availability_multiplier, axis=1)
    shortlist_df["score"] = [
        apply_availability_multiplier(s, m)
        for s, m in zip(shortlist_df["model_score"], shortlist_df["availability_multiplier"])
    ]
    print(
        f"Availability modifier applied: mean {shortlist_df['availability_multiplier'].mean():.3f}, "
        f"{(shortlist_df['availability_multiplier'] < 0.8).sum()} of {len(shortlist_df)} shortlisted candidates below 0.8."
    )

    print("Stage 7: Filtering out honeypots...")
    shortlist_df = pd.merge(shortlist_df, df_hp, on="candidate_id")
    non_hp_df = shortlist_df[shortlist_df["is_honeypot"] == False].copy()
    print(f"Candidates remaining after honeypot filtering: {len(non_hp_df)}.")

    print("Stage 8: Sorting, selecting top 100, and formatting output...")
    # Clean score ties with alphabetical candidate_id sort
    # Extracts the numeric part of the ID for robust sorting
    non_hp_df["candidate_id_num"] = non_hp_df["candidate_id"].str.extract(r"(\d+)").astype(int)
    non_hp_df = non_hp_df.sort_values(by=["score", "candidate_id_num"], ascending=[False, True])

    top_100 = non_hp_df.head(100).copy()
    top_100["rank"] = range(1, len(top_100) + 1)

    print("Stage 9: Generating reasoning for finalists...")
    if args.reasoning_mode == "grounded-llm":
        import json as _json
        from reasoning_llm_rewrite import generate_reasonings_llm_rewrite
        diag = []
        top_100["reasoning"] = generate_reasonings_llm_rewrite(top_100, diagnostics=diag, model_name=args.llm_model, attempts=1, cache_path=args.rewrite_cache)
        if args.reasoning_diagnostics:
            args.reasoning_diagnostics.write_text(_json.dumps(diag, indent=1))
    elif args.reasoning_mode == "grounded-cached":
        import json as _json
        from reasoning_llm_rewrite import generate_reasonings_cached
        diag = []
        top_100["reasoning"] = generate_reasonings_cached(top_100, args.rewrite_cache, diagnostics=diag)
        if args.reasoning_diagnostics:
            args.reasoning_diagnostics.write_text(_json.dumps(diag, indent=1))
    elif args.reasoning_mode == "slotted-cached":
        import json as _json
        from reasoning_slotted import generate_reasonings_slotted_cached
        diag = []
        top_100["reasoning"] = generate_reasonings_slotted_cached(top_100, args.rewrite_cache, diagnostics=diag)
        if args.reasoning_diagnostics:
            args.reasoning_diagnostics.write_text(_json.dumps(diag, indent=1))
    elif args.reasoning_mode == "llm":
        from reasoning import generate_reasonings  # legacy path; imports torch/transformers lazily
        top_100["reasoning"] = generate_reasonings(top_100)
    else:
        top_100["reasoning"] = generate_reasonings_grounded(top_100)

    # Output CSV format: candidate_id, rank, score, reasoning
    final_output = top_100[["candidate_id", "rank", "score", "reasoning"]]
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_output.to_csv(args.out, index=False)
    print(f"Stage 10: Wrote final ranking list of {len(final_output)} rows to {args.out}")
    print(f"Total pipeline execution time: {time.time() - pipeline_start_time:.2f}s")


if __name__ == "__main__":
    main()
