# Tests for the slot-based LLM reasoning (reasoning_slotted.py). No model is loaded: templates are supplied by hand.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from reasoning_grounded import build_facts, build_slot_parts, compose_reasoning  # noqa: E402
from reasoning_slotted import (  # noqa: E402
    build_messages, fill_slots, gate_slotted, generate_reasonings_slotted, generate_reasonings_slotted_cached)
from reasoning_llm_rewrite import draft_key, load_cache  # noqa: E402
from test_reasoning_grounded import _row  # noqa: E402

import pandas as pd  # noqa: E402


def _setup(rank=3, cid="CAND_0000001", **over):
    r = _row(cid=cid, **over)
    f = build_facts(r)
    return r, build_slot_parts(f, rank), compose_reasoning(f, rank)


def _good(spec):
    """A hand-made valid template for whichever slots this row has."""
    p = spec["parts"]
    s1 = "[WHO] has [YEARS] and is a [FIT]" + (", listing [SKILLS]" if "SKILLS" in p else "") + (" and a role described as [QUOTE]" if "QUOTE" in p else "") + "."
    if "CONCERN1" in p:
        s2 = "The main concern is [CONCERN1]" + (" and [CONCERN2]" if "CONCERN2" in p else "") + "."
    else:
        s2 = "Availability: [AVAIL]." if "NOTE" not in p else "[NOTE]. Availability: [AVAIL]."
    return f"{s1} {s2}"


def test_slot_parts_are_verified_facts_and_required_set_is_consistent():
    r, spec, _ = _setup()
    assert spec["parts"]["WHO"] == f"{build_facts(r).title} at {build_facts(r).company}"
    assert set(spec["required"]) <= set(spec["parts"]) and "PRODUCT" not in spec["required"] and "BAND" not in spec["required"]


def test_good_template_is_accepted_and_filled_with_the_exact_facts():
    r, spec, d = _setup()
    t = _good(spec)
    assert gate_slotted(t, spec, r, 3, d) == []
    text = fill_slots(t, spec)
    assert "[" not in text and "]" not in text
    assert all(v in text for k, v in spec["parts"].items() if k in spec["required"])


def test_missing_duplicate_and_unknown_placeholders_are_rejected():
    r, spec, d = _setup()
    t = _good(spec)
    assert any("missing" in x for x in gate_slotted(t.replace("[YEARS]", "many years"), spec, r, 3, d))
    assert any("more than once" in x for x in gate_slotted(t + " [FIT]", spec, r, 3, d))
    assert any("unknown placeholder" in x for x in gate_slotted(t.replace("[YEARS]", "[SALARY]"), spec, r, 3, d))


def test_model_cannot_write_facts_itself():
    r, spec, d = _setup()
    t = _good(spec)
    assert any("digits" in x for x in gate_slotted(t.replace("has [YEARS]", "has 9 years and [YEARS]"), spec, r, 3, d))
    assert any("quotation" in x for x in gate_slotted(t.replace("and is a", 'and is a "very" and a'), spec, r, 3, d))
    assert any("names" in x or "outside plain" in x for x in gate_slotted(t.replace("and is a", "and, like Google, is a"), spec, r, 3, d))


def test_praise_and_job_posting_wording_are_rejected():
    r, spec, d = _setup()
    t = _good(spec)
    assert any("evaluation" in x for x in gate_slotted(t.replace("is a", "is an excellent"), spec, r, 3, d))
    assert any("meaning" in x for x in gate_slotted(t.replace("has [YEARS]", "is offering a position with [YEARS]"), spec, r, 3, d))
    assert any("rank" in x for x in gate_slotted(t.replace("and is a", "and by rank is a"), spec, r, 3, d))


def test_words_outside_plain_connectives_are_rejected():
    r, spec, d = _setup()
    t = _good(spec)
    bad = gate_slotted(t.replace("and is a", "and is currently quite plausibly a"), spec, r, 3, d)
    assert any("outside plain" in x for x in bad)


def test_concern_must_be_framed_as_a_concern_and_not_invented():
    r, spec, d = _setup()
    if "CONCERN1" in spec["parts"]:
        assert any("not framed" in x for x in gate_slotted(_good(spec).replace("The main concern is", "It also has"), spec, r, 3, d))
    else:
        assert any("no concern" in x for x in gate_slotted(_good(spec) + " The gap is small.", spec, r, 3, d))


def test_article_and_grammar_slips_are_rejected_and_articles_repaired():
    r, spec, d = _setup()
    t = _good(spec)
    assert any("article" in x for x in gate_slotted(t.replace("has [YEARS]", "has a [YEARS]"), spec, r, 3, d))
    assert any("[FIT]" in x for x in gate_slotted(t.replace("is a [FIT]", "is [FIT]"), spec, r, 3, d))
    assert fill_slots("This is an [FIT].", {"parts": {"FIT": "solid match"}}) == "This is a solid match."
    assert fill_slots("This is a [FIT].", {"parts": {"FIT": "average match"}}) == "This is an average match."


def test_wrong_subject_is_rejected():
    r, spec, d = _setup()
    t = "The company hired [WHO]; [YEARS], a [FIT], [SKILLS]. The concern is [CONCERN1]." if "CONCERN1" in spec["parts"] else "The company hired [WHO]. [YEARS], a [FIT], [SKILLS]. Availability: [AVAIL]."
    assert gate_slotted(t, spec, r, 3, d)


def test_build_messages_shows_slots_not_facts():
    r, spec, d = _setup()
    last = build_messages(spec)[-1]["content"]
    assert "[WHO]" in last and spec["parts"]["WHO"] not in last and "MEANINGS" in last


def test_generation_falls_back_to_the_draft_when_the_template_is_bad_and_fills_when_good(tmp_path):
    rows = [_row(cid=f"CAND_{i:07d}") for i in range(1, 4)]
    df = pd.DataFrame(rows)
    df["rank"] = [1, 2, 3]
    drafts = [compose_reasoning(build_facts(r), k) for r, k in zip(rows, [1, 2, 3])]

    def gen(msgs):  # the first row gets an unusable template (no placeholders), the others get a fully required-slot template
        outs = []
        for i, m in enumerate(msgs):
            req = [x for x in m if x["role"] == "user"][-1]["content"].split("\n")[0].replace("REQUIRED:", "").split()
            outs.append("This candidate is fine." if i == 0 and len(msgs) == 3 else _tmpl_for(req))
        return outs
    diag = []
    cache = tmp_path / "c.json"
    out = generate_reasonings_slotted(df, diagnostics=diag, generator=gen, attempts=1, cache_path=cache)
    assert out[0] == drafts[0] and not diag[0]["accepted"]
    cached = generate_reasonings_slotted_cached(df, cache)
    for i in (1, 2):
        if diag[i]["accepted"]:
            assert cached[i] == out[i] and draft_key(drafts[i]) in load_cache(cache)
    assert cached[0] == drafts[0]


def _tmpl_for(req):
    s1 = "[WHO] has [YEARS] and is a [FIT]" + (", listing [SKILLS]" if "[SKILLS]" in req else "") + (" and a role described as [QUOTE]" if "[QUOTE]" in req else "") + "."
    if "[CONCERN1]" in req:
        s2 = "The main concern is [CONCERN1]" + (" and [CONCERN2]" if "[CONCERN2]" in req else "") + "."
    else:
        s2 = ("[NOTE]. " if "[NOTE]" in req else "") + "Availability: [AVAIL]."
    return f"{s1} {s2}"


def test_cached_template_is_refilled_from_the_current_record_not_trusted_as_text(tmp_path):
    r = _row(cid="CAND_0000001")
    df = pd.DataFrame([r]); df["rank"] = [3]
    cache = tmp_path / "c.json"
    f = build_facts(r); spec = build_slot_parts(f, 3); d = compose_reasoning(f, 3)
    t = _tmpl_for([f"[{k}]" for k in spec["required"]])
    if gate_slotted(t, spec, r, 3, d):  # template shape not valid for this fixture row: nothing to test
        return
    cache.write_text(json.dumps({draft_key(d): {"template": t, "rewrite": "STALE TEXT", "mode": "slotted"}}), encoding="utf-8")
    out = generate_reasonings_slotted_cached(df, cache)
    assert out[0] != "STALE TEXT" and spec["parts"]["WHO"] in out[0]


def test_positive_fact_inside_a_concern_sentence_is_rejected():
    r, spec, d = _setup()
    t = _good(spec)
    assert any("positive fact" in y for y in gate_slotted(t.replace("The main concern is [CONCERN1]", "The concern is a role described as [QUOTE] and [CONCERN1]").replace("and [QUOTE].", ".")
                                                          if "CONCERN1" in spec["parts"] else t.replace("Availability: [AVAIL]", "The gap is [QUOTE] and [AVAIL]").replace(" and a role described as [QUOTE].", "."), spec, r, 3, d))


def test_bare_quote_needs_an_introduction_and_years_needs_no_of():
    r, spec, d = _setup()
    t = _good(spec)
    if "QUOTE" in spec["parts"]:
        assert any("introduce [QUOTE]" in y for y in gate_slotted(t.replace("and a role described as [QUOTE]", "and [QUOTE]"), spec, r, 3, d))
    assert any("of experience" in y for y in gate_slotted(t.replace("has [YEARS]", "has [YEARS] of"), spec, r, 3, d))


def test_diversity_pick_prefers_the_least_used_skeleton_and_skips_rows_without_candidates():
    from reasoning_slotted import _pick_diverse
    a, b = "[WHO] is a [FIT].", "With [YEARS], [WHO] is a [FIT]."
    picks = _pick_diverse([[(1, a)], [(1, a), (2, b)], [], [(1, a), (3, b)]])  # usage after rows 0-1: a=1, b=1
    assert picks[0] == (1, a) and picks[1] == (2, b) and picks[2] is None and picks[3][1] == a  # a tie keeps the earliest
