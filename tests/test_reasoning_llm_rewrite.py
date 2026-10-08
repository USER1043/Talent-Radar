# Tests for the LLM-rewrite gate and prompt plumbing (reasoning_llm_rewrite.py). No model is loaded: rewrites are supplied by hand.
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from reasoning_grounded import build_facts, compose_reasoning  # noqa: E402
from reasoning_llm_rewrite import (  # noqa: E402
    _clean, build_messages, draft_key, gate_rewrite, generate_reasonings_cached, generate_reasonings_llm_rewrite, keep_list, load_cache, prefill_phrase, save_cache)
from test_reasoning_grounded import _row  # noqa: E402


def _draft(rank=3, cid="CAND_0000001", **over):
    r = _row(cid=cid, **over)
    return r, compose_reasoning(build_facts(r), rank)


def _draft_with(phrase, rank=3, **over):
    """A draft (and its row) that contains `phrase`; the frame is picked by a hash of the id, so scan a few ids."""
    for i in range(1, 200):
        r, d = _draft(rank, cid=f"CAND_{i:07d}", **over)
        if phrase in d:
            return r, d
    raise AssertionError(f"no draft with {phrase!r}")


def _faithful(draft):
    """A hand-made rewrite that changes wording but keeps every fact."""
    return draft.replace("Main concern:", "The one concern is").replace("The gap to watch:", "The one concern is").replace("Caveat:", "The one concern is")


def _draft_of(messages):
    last_user = [m for m in messages if m["role"] == "user" and m["content"].startswith("DRAFT:")][0]  # the row's own first DRAFT message
    users = [m for m in messages if m["role"] == "user" and m["content"].startswith("DRAFT:")]
    return re.search(r"DRAFT: (.*)\nKEEP:", users[-1]["content"], re.S).group(1)


def test_gate_accepts_an_unchanged_or_faithfully_reworded_draft():
    r, d = _draft()
    assert gate_rewrite(d, d, r, 3) == []
    assert gate_rewrite(_faithful(d), d, r, 3) == []


def test_gate_allows_dropping_optional_positives():
    r, d = _draft_with("product-company experience", redrob_notice_period_days=120)
    dropped = d.replace(" and product-company experience", "").replace("product-company experience and ", "")
    assert dropped != d and gate_rewrite(dropped, d, r, 3) == []
    r2, d2 = _draft_with("experience inside the JD's 5-9 year band", redrob_notice_period_days=120)
    dropped2 = d2.replace(" and experience inside the JD's 5-9 year band", "").replace("experience inside the JD's 5-9 year band and ", "")
    assert dropped2 != d2 and gate_rewrite(dropped2, d2, r2, 3) == []


def test_gate_rejects_each_kind_of_unfaithful_rewrite():
    r, d = _draft(redrob_notice_period_days=120)
    facts = build_facts(r)
    claims = [f"{s['name']} ({s['proficiency']}, {s['months']} mo)" for s in facts.skills_named]
    bad = {
        "dropped two skills": d.replace(claims[0], "a vector database").replace(claims[1], "a ranking model"),
        "altered months": d.replace(f"{facts.skills_named[0]['months']} mo", f"{facts.skills_named[0]['months'] + 5} mo"),
        "invented skill": d.replace(claims[0], "Kubeflow (expert, 40 mo)"),
        "new number": d.rstrip(".") + " and 12 publications.",
        "invented employer": d.replace(facts.company, "Google", 1),
        "role swap (company as subject)": f"{facts.company} offers a strong fit for a {facts.yoe:g}-year {facts.title} with {claims[0]} and {claims[1]}. The notice period is 120 days.",
        "dropped concern": re.split(r" (?:Main concern|The gap to watch|Caveat|Held back by)", d)[0],
        "rank mention": "Ranked #3 overall. " + d,
        "first person": d + " I think they are great.",
        "new capitalised name": d.rstrip(".") + ", similar to work at Anthropic.",
        "way too short": "Good candidate.",
        "invented quote": d.rstrip(".") + ' and a role described as "Led search at Google".',
    }
    for name, text in bad.items():
        assert gate_rewrite(text, d, r, 3), f"gate accepted an unfaithful rewrite: {name}"


def test_gate_rejects_meaning_drift_wording():
    r, d = _draft(redrob_notice_period_days=120)
    for name, text in {
        "job-posting framing": d.rstrip(".") + ", and the company is recruiting for this position.",
        "fact turned into a requirement": d.replace("120-day notice period", "120-day notice period is required", 1) if "120-day notice period" in d else d + " A notice period is required.",
        "praise upgrade": d.replace("skills", "expertise in skills", 1),
        "grammar defect": d.rstrip(".") + ", holding back by the notice period.",
    }.items():
        assert any("meaning" in x for x in gate_rewrite(text, d, r, 3)), f"drift not caught: {name}"


def test_keep_list_carries_the_verdict_title_company_skills_and_concerns():
    r, d = _draft(redrob_notice_period_days=120)
    keep = keep_list(d, r)
    f = build_facts(r)
    for must in (f.title, f.company, f"{f.yoe:g} years", f"{f.skills_named[0]['name']} ({f.skills_named[0]['proficiency']}, {f.skills_named[0]['months']} mo)", "120-day notice period"):
        assert must in keep, f"{must!r} missing from KEEP"
    assert any(v in keep for v in ("strong fit", "clear fit", "very good match", "good fit", "solid match", "good match", "reasonable fit", "moderate match", "borderline fit", "partial match", "marginal match", "partial-to-good match"))
    msgs = build_messages(d, r)
    assert msgs[0]["role"] == "system" and "candidate is always the subject" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user" and d in msgs[-1]["content"] and "KEEP:" in msgs[-1]["content"]


def test_prefill_forces_the_correct_who_works_where_relation():
    r, d = _draft()
    f = build_facts(r)
    phrase = prefill_phrase(r, 3)
    assert f.title in phrase and f.company in phrase
    # whatever continuation the model writes after the forced opening, the relation check is satisfied
    assert not any("who works where" in x for x in gate_rewrite(phrase + " has good skills.", d, r, 3))
    seen = []
    df = pd.DataFrame([r]); df["rank"] = [3]
    generate_reasonings_llm_rewrite(df, generator=lambda ms: seen.append(ms) or ["x"] * len(ms), attempts=1)
    assert seen[0][0][-1] == {"role": "assistant", "content": phrase, "prefill": True}


def test_gate_rejects_claims_the_draft_never_made():
    r, d = _draft(redrob_notice_period_days=30, redrob_open_to_work_flag=True)
    assert "open-to-work" not in d
    assert gate_rewrite(d.rstrip(".") + ", and there is no open-to-work flag.", d, r, 3)      # a concern leaked from an example
    assert gate_rewrite(d.replace("82% recruiter response rate", "high 82% recruiter response rate"), d, r, 3) if "82% recruiter response rate" in d else True
    facts = build_facts(r)
    c = facts.skills_named[0]
    assert gate_rewrite(d.replace(f"{c['name']} ({c['proficiency']}, {c['months']} mo)", f"{c['name'].lower()} {c['proficiency']}, {c['months']} mo"), d, r, 3)  # skill without its exact form
    assert gate_rewrite(d + " " + d.split(".")[0] + ".", d, r, 3)                               # repeats itself
    assert any("evaluation" in x for x in gate_rewrite(d.replace("skills", "excellent skills", 1), d, r, 3))
    assert gate_rewrite(d.rstrip(".") + ". The role is located in Chennai.", d, r, 3)             # the role's location is not the candidate's


def test_gate_enforces_tone_by_rank_tier():
    r, d_top = _draft(rank=2)
    assert gate_rewrite(d_top.replace("fit", "fit and a borderline case", 1), d_top, r, 2)  # weak wording on a top-10 row
    r2, d_low = _draft(rank=90)
    assert gate_rewrite(d_low + " An outstanding profile.", d_low, r2, 90)  # strong wording on a rank>70 row


def test_gate_requires_the_rank_tier_verdict_to_survive():
    r, d = _draft(rank=90)
    verdict = next(v for v in ("borderline fit", "partial match", "marginal match") if v in d.lower())
    assert gate_rewrite(d.replace(verdict, "an option", 1).replace(verdict.title(), "An option", 1), d, r, 90)


def test_fallback_to_draft_when_llm_output_fails_or_generator_crashes():
    rows = [_row(cid=f"CAND_{i:07d}") for i in range(3)]
    df = pd.DataFrame(rows)
    df["rank"] = [1, 2, 3]
    drafts = [compose_reasoning(build_facts(r), i + 1) for i, r in enumerate(rows)]
    diag = []
    out = generate_reasonings_llm_rewrite(df, diagnostics=diag, generator=lambda ms: ["Ranked #1 because reasons."] * len(ms))
    assert out == drafts and not any(x["accepted"] for x in diag)  # every bad rewrite falls back to its draft

    def crash(_):
        raise RuntimeError("model not cached")
    assert generate_reasonings_llm_rewrite(df, generator=crash) == drafts

    diag = []
    out = generate_reasonings_llm_rewrite(df, diagnostics=diag, generator=lambda ms: [_faithful(_draft_of(m)) for m in ms])
    assert all(x["accepted"] and x["attempt"] == 1 for x in diag) and out == [_faithful(d) for d in drafts]


def test_repair_pass_feeds_the_gate_complaints_back_and_can_rescue_a_row():
    rows = [_row(cid="CAND_0000042")]
    df = pd.DataFrame(rows)
    df["rank"] = [3]
    draft = compose_reasoning(build_facts(rows[0]), 3)
    seen = []

    def generator(msgs):
        seen.append(msgs)
        if len(seen) == 1:
            return ["Ranked #3 because reasons."]  # bad first attempt
        complaint = [m for m in msgs[0] if m["role"] == "user" and "not acceptable" in m["content"]]
        assert complaint and "Ranked" in complaint[0]["content"]  # the gate's complaint is fed back
        return [_faithful(draft)]  # good second attempt

    diag = []
    out = generate_reasonings_llm_rewrite(df, diagnostics=diag, generator=generator)
    assert len(seen) == 2 and out == [_faithful(draft)] and diag[0]["accepted"] and diag[0]["attempt"] == 2


def test_clean_strips_wrappers():
    assert _clean('REWRITE: "A short note that is fine."') == "A short note that is fine."
    assert _clean("First sentence. Second sentence that is cut off") == "First sentence."


def test_cache_roundtrip_and_cached_mode_uses_only_gate_verified_rewrites(tmp_path):
    rows = [_row(cid=f"CAND_{i:07d}") for i in range(4)]
    df = pd.DataFrame(rows)
    df["rank"] = [1, 2, 3, 4]
    drafts = [compose_reasoning(build_facts(r), i + 1) for i, r in enumerate(rows)]
    cache_file = tmp_path / "cache.json"
    cache = {
        draft_key(drafts[0]): {"rewrite": _faithful(drafts[0]), "model": "test"},                 # good -> used
        # drafts[1]: no entry at all                                                              # miss -> draft
        draft_key(drafts[2]): {"rewrite": drafts[2].replace("years", "yrs", 1), "model": "test"},  # fails the gate (years wrong/missing) -> draft
        draft_key(drafts[3]): {"rewrite": "Ranked #4 because reasons.", "model": "test"},          # fails the gate -> draft
    }
    save_cache(cache_file, cache)
    assert load_cache(cache_file) == cache
    diag = []
    out = generate_reasonings_cached(df, cache_file, diagnostics=diag)
    assert out[0] == _faithful(drafts[0]) and out[1:] == drafts[1:]
    assert [d["accepted"] for d in diag] == [True, False, False, False] and diag[1]["problems"] == ["cache miss"]
    assert generate_reasonings_cached(df, tmp_path / "missing.json") == drafts  # no cache file at all: plain grounded text


def test_live_mode_writes_accepted_rewrites_into_the_cache(tmp_path):
    rows = [_row(cid=f"CAND_{i:07d}") for i in range(2)]
    df = pd.DataFrame(rows)
    df["rank"] = [1, 2]
    drafts = [compose_reasoning(build_facts(r), i + 1) for i, r in enumerate(rows)]
    cache_file = tmp_path / "cache.json"
    outs = {drafts[0]: _faithful(drafts[0]), drafts[1]: "Ranked #2 because reasons."}
    generate_reasonings_llm_rewrite(df, generator=lambda ms: [outs[_draft_of(m)] for m in ms], attempts=1, cache_path=cache_file)
    saved = load_cache(cache_file)
    assert set(saved) == {draft_key(drafts[0])} and saved[draft_key(drafts[0])]["rewrite"] == _faithful(drafts[0])  # only the accepted one is kept


def test_the_verbatim_quote_must_be_kept_when_the_draft_has_one():
    r, d = _draft_with('a role described as "', redrob_notice_period_days=120)
    assert gate_rewrite(d, d, r, 3) == []
    dropped = re.sub(r' (?:and )?a role described as "[^"]+"( and)?', "", d)
    assert dropped != d and any("quote" in p for p in gate_rewrite(dropped, d, r, 3))
    assert "QUOTE RULE" in build_messages(d, r)[-1]["content"]


def test_gate_rejects_the_a_n_year_experience_grammar_slip():
    r, d = _draft(redrob_notice_period_days=120)
    assert any("grammar" in p for p in gate_rewrite(d.rstrip(".") + ". The candidate has a 3-year experience below the band.", d, r, 3))
