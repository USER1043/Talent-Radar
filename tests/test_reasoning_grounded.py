# Tests for the fact-grounded reasoning generator (reasoning_grounded.py).
# Self-contained: builds synthetic candidate rows, so no data files are needed.
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from reasoning_grounded import (  # noqa: E402
    audit_reasoning,
    build_facts,
    compose_reasoning,
    generate_reasonings_grounded,
)


def _row(cid="CAND_0000001", **over):
    base = dict(
        candidate_id=cid, current_title="Senior ML Engineer", current_company="Swiggy", years_of_experience=6.5,
        country="India", location="Bengaluru, Karnataka", n_employers=3, avg_tenure_months=30.0,
        has_product_company_experience=True, is_consulting_only=False,
        narrative_text="Built the ranking and evaluation stack with NDCG dashboards.",
        skills_json=json.dumps([
            {"name": "Qdrant", "proficiency": "expert", "endorsements": 9, "duration_months": 60},
            {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 4, "duration_months": 40},
            {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 90},
            {"name": "Photoshop", "proficiency": "expert", "endorsements": 1, "duration_months": 30},
        ]),
        career_history_json=json.dumps([
            {"company": "Swiggy", "title": "Senior ML Engineer", "is_current": True, "duration_months": 30,
             "description": "Owned search ranking end to end. Wrote the weekly report."},
        ]),
        redrob_recruiter_response_rate=0.82, redrob_last_active_date="2026-06-01", redrob_notice_period_days=30,
        redrob_open_to_work_flag=True, redrob_github_activity_score=55.0, redrob_willing_to_relocate=True,
    )
    base.update(over)
    return pd.Series(base)


def test_grounded_text_passes_audit_and_is_specific():
    r = _row()
    t = compose_reasoning(build_facts(r), 3)
    assert audit_reasoning(t, r) == []
    assert "Senior ML Engineer" in t and "Swiggy" in t and "6.5" in t
    assert "Qdrant (expert, 60 mo)" in t
    assert "Photoshop" not in t  # not a JD skill, must never be surfaced
    assert not t.startswith("Ranked")


def test_only_owned_skills_are_named():
    r = _row(skills_json=json.dumps([{"name": "Python", "proficiency": "expert", "endorsements": 1, "duration_months": 70}]))
    t = compose_reasoning(build_facts(r), 5)
    assert "Qdrant" not in t and "Learning to Rank" not in t
    assert "no listed retrieval or vector-search skill" in t  # honest gap instead of a made-up strength


def test_concerns_are_real_and_specific():
    r = _row(redrob_recruiter_response_rate=0.07, redrob_notice_period_days=120, redrob_last_active_date="2025-10-01",
             redrob_open_to_work_flag=False, country="Canada", location="Toronto, Ontario")
    t = compose_reasoning(build_facts(r), 40)
    # mid-list rows report the two most severe concerns: unresponsiveness first, then inactivity
    assert "low recruiter response rate (7%)" in t
    assert "months of inactivity (last active 2025-10-01)" in t
    assert audit_reasoning(t, r) == []
    # a top-10 row reports just the single most severe concern, and it must be a real one
    t1 = compose_reasoning(build_facts(r), 5)
    assert "low recruiter response rate (7%)" in t1 and "inactivity" not in t1

    # a long notice period is surfaced when it is the most severe issue
    r2 = _row(redrob_notice_period_days=120)
    t2 = compose_reasoning(build_facts(r2), 20)
    assert "120-day notice period against the JD's preference for under 30" in t2 and audit_reasoning(t2, r2) == []


def test_no_concern_case_is_honest_not_invented():
    r = _row()
    t = compose_reasoning(build_facts(r), 2)
    assert "82% recruiter response rate" in t and "30-day notice" in t
    # a low-ranked row with nothing to flag must not sound glowing: it says the rank comes from the relative score
    low = compose_reasoning(build_facts(r), 90)
    assert "relative score" in low and "82% recruiter response rate" in low


def test_tone_follows_rank():
    r = _row()
    top = compose_reasoning(build_facts(r), 1).lower()
    bottom = compose_reasoning(build_facts(r), 100).lower()
    assert any(w in top for w in ("strong fit", "clear fit", "very good match"))
    assert any(w in bottom for w in ("borderline fit", "partial match", "marginal match"))
    assert not any(w in bottom for w in ("strong fit", "clear fit", "very good match"))


def test_deterministic():
    r = _row()
    assert compose_reasoning(build_facts(r), 7) == compose_reasoning(build_facts(r), 7)


def test_many_candidates_are_all_distinct_and_grounded():
    rows = [_row(cid=f"CAND_{i:07d}", years_of_experience=4 + (i % 9) * 0.7,
                 redrob_recruiter_response_rate=0.1 + (i % 9) * 0.1, redrob_notice_period_days=[0, 15, 30, 60, 90, 120][i % 6],
                 current_company=["Swiggy", "Ola", "Razorpay", "CRED"][i % 4]) for i in range(24)]
    df = pd.DataFrame(rows)
    df["rank"] = range(1, 25)
    texts = generate_reasonings_grounded(df)
    assert len(set(texts)) == len(texts)
    assert all(audit_reasoning(t, r) == [] for t, (_, r) in zip(texts, df.iterrows()))
    assert not any(t.startswith("Ranked") for t in texts)


def test_audit_catches_fabrication():
    r = _row()
    t = compose_reasoning(build_facts(r), 3)
    bad = {
        "fake skill": re.sub(r"skills including", "skills including Kubeflow (expert, 40 mo),", t, count=1),
        "wrong months": re.sub(r"\((\w+), (\d+) mo\)", lambda m: f"({m.group(1)}, {int(m.group(2)) + 9} mo)", t, count=1),
        "wrong years": t.replace("6.5", "11", 1),
        "invented quote": t.rstrip(".") + ' and a role described as "Led search at Google".',
        "generic opener": "Ranked #3 because " + t,
    }
    for name, text in bad.items():
        assert audit_reasoning(text, r), f"audit missed: {name}"
