"""
@file Slot-based LLM reasoning (`--reasoning-mode slotted-cached`, offline build with `slotted-llm`).
@package online_ranking

The LLM writes ONLY the connecting words. Every fact is a placeholder such as [WHO], [SKILLS] or [QUOTE] that the code
replaces afterwards with text taken from the candidate's own record, so the model cannot drop, reword or invent a fact.

  1. reasoning_grounded.build_slot_parts() gives the verified text for each slot of this candidate;
  2. the model writes a 2-sentence template using each required slot exactly once;
  3. `gate_slotted` accepts the template only if the slots are exactly right and the words OUTSIDE the slots are harmless:
     no digits, quotes or names, no praise or job-posting wording, no words beyond plain connectives and the row's own
     draft vocabulary, concerns framed as concerns;
  4. the code fills the slots; the filled text must also pass the record audit;
  5. a row that fails gets one repair attempt, then falls back to the deterministic grounded draft.
The cache stores the TEMPLATE; at ranking time it is re-gated and re-filled from the current record (no model needed).
"""

from __future__ import annotations

import re

import pandas as pd

from reasoning_grounded import audit_reasoning, build_facts, build_slot_parts, compose_reasoning
from reasoning_llm_rewrite import (DEFAULT_MODEL, _DRIFT, _EVALUATIVE, _FIRST_PERSON, _NEGATIVE, _POSITIVE, _clean, _generate, _relation_ok,
                                   _verify_skill_triples, draft_key, load_cache, save_cache)

SLOT_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")

_LEGEND = {
    "WHO": "the job title and employer, one noun phrase (it already contains the word 'at'). It must be the grammatical subject",
    "YEARS": "the years of experience, a noun phrase that already ends with 'of experience' (use it like 'with [YEARS]')",
    "FIT": "the fit verdict, a noun phrase (use it like 'is a [FIT]')",
    "SKILLS": "a list of the candidate's skills (use it like 'listing [SKILLS]')",
    "QUOTE": "a quotation, in double quotes, from the candidate's own role description. Introduce it yourself, like 'a role described as [QUOTE]'",
    "PRODUCT": "a noun phrase about product-company experience",
    "CONCERN1": "the first concern, a noun phrase",
    "CONCERN2": "the second concern, a noun phrase",
    "AVAIL": "availability facts, a noun phrase list",
    "NOTE": "a full clause that already ends the thought; use it as the whole second sentence or after a colon",
}

# plain grammar words the model may use outside the slots, on top of the words already in the row's own grounded draft
_FUNCTION = set("""a an the and or but so of at in on for with by from to as into over under about against than that which who whom whose this these those it its
he she they his her their them there here is are was were be been being am do does did has have had having also plus while though although however yet still only
just both either neither not no nor per each every all any some one other another such more most less least much many few can could may might will would should
must when where what how why if then because since after before during between within without across along around through toward towards up down out off
listing lists listed list shows show showing shown includes include included including brings bring bringing comes come coming carries carry combines adds add
notes note noted worth watch point points key main gap gaps concern concerns caveat profile candidate background record history skills skill role
primary major biggest additionally experience description described describes describe describing states says said mentioned mentions reads role roles
""".split())
_CONCERN_WORDS = ("concern", "gap", "caveat", "downside", "drawback", "risk", "weakness", "held back", "holds back", "shortfall")
_CONCERN_CUES = _CONCERN_WORDS + ("watch", "but", "however", "although", "though", "while", "offset", "against", "limited", "lack", "flag",
                                  "tempered", "tempers", "counts against", "weighs", "weigh", "still", "yet", "only", "worry", "issue")
_NAME_FREE_OK = {"jd"}
_POSITIVE_SLOTS = ("WHO", "YEARS", "FIT", "SKILLS", "QUOTE", "PRODUCT")

_SYSTEM = (
    "You write two short sentences about ONE job candidate for a recruiter. All facts are supplied as placeholders in square brackets, "
    "and you write only the connecting words around them.\n"
    "Rules:\n"
    "- Use every REQUIRED placeholder exactly once, written exactly like [WHO]. You may also use an OPTIONAL placeholder, at most once.\n"
    "- The candidate is the subject. Start with [WHO] or use it early.\n"
    "- Write no digits, no names, no skills, no double quotes. Never write a fact yourself: use the placeholder instead.\n"
    "- Add no praise or criticism words of your own; the [FIT] placeholder carries the verdict. Never mention a rank or say that a company hires or offers.\n"
    "- Frame a [CONCERN1] or [CONCERN2] placeholder clearly as a concern (for example 'The main concern is [CONCERN1]').\n"
    "- Write exactly 2 sentences using plain, simple words."
)
# (has_concern, user request, assistant template): different sentence shapes on purpose, so the model does not copy one skeleton
_EXAMPLES = [
    (True, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [CONCERN1]\nOPTIONAL: [PRODUCT]",
     "[WHO] brings [YEARS] and is a [FIT], listing [SKILLS] along with [PRODUCT]. The one gap to watch is [CONCERN1]."),
    (True, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [QUOTE] [CONCERN1] [CONCERN2]",
     "With [YEARS], [WHO] is a [FIT], listing [SKILLS] and a role described as [QUOTE]. The main concerns are [CONCERN1] and [CONCERN2]."),
    (True, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [QUOTE] [CONCERN1]\nOPTIONAL: [PRODUCT]",
     "[WHO] is a [FIT] with [YEARS]. The profile shows [SKILLS], and the role description reads [QUOTE]; the concern is [CONCERN1]."),
    (True, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [CONCERN1] [CONCERN2]",
     "A [FIT]: [WHO], with [YEARS] and [SKILLS]. Concerns to note: [CONCERN1], and also [CONCERN2]."),
    (False, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [AVAIL]\nOPTIONAL: [PRODUCT]",
     "[WHO] has [YEARS] and is a [FIT], with [SKILLS]. Availability: [AVAIL]."),
    (False, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [QUOTE] [AVAIL]\nOPTIONAL: [PRODUCT]",
     "[WHO] is a [FIT] with [YEARS]. The profile lists [SKILLS], with a role described as [QUOTE], and availability is [AVAIL]."),
    (False, "REQUIRED: [WHO] [YEARS] [FIT] [SKILLS] [AVAIL] [NOTE]",
     "With [YEARS], [WHO] is a [FIT], listing [SKILLS]. On the record, [NOTE]; availability: [AVAIL]."),
]
_OPENERS = ("[WHO]", "With [YEARS],", "A [FIT]:")  # forced start of the answer, rotated per candidate


def _seed_of(row: pd.Series) -> int:
    return int(re.sub(r"\D", "", str(row["candidate_id"])) or 0)


def _examples_for(spec: dict, seed: int) -> list[tuple[str, str]]:
    """Three worked examples matching this row's kind (concern / no concern), rotated by candidate so the LAST one (the one the model imitates most) varies."""
    kind = any(k.startswith("CONCERN") for k in spec["parts"])
    pool = [(u, a) for c, u, a in _EXAMPLES if c == kind]
    start = seed % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(min(3, len(pool)))]


# ---------------------------------------------------------------- slot handling
def _article_for(value: str) -> str:
    return "an" if re.match(r"[aeiouAEIOU]", value.strip()) and not re.match(r"(?:uni|use|eu)", value.strip(), re.I) else "a"


def fill_slots(template: str, spec: dict) -> str:
    """Deterministically replace each [SLOT] with its verified text and repair 'a'/'an' before a slot."""
    parts = spec["parts"]
    template = re.sub(r"\[YEARS\]\s+(?:of\s+)?experience\b", "[YEARS]", template)  # the slot already ends with "of experience"
    t = re.sub(r"\b([Aa]n?) \[([A-Za-z0-9_]+)\]", lambda m: _art(m, parts), template)
    def value(m: re.Match) -> str:
        v = parts.get(m.group(1), m.group(0))
        before = t[: m.start()]
        return v[:1].upper() + v[1:] if not before.strip() or re.search(r"[.!?]\s*$", before) else v
    t = SLOT_RE.sub(value, t)
    return re.sub(r"\s+", " ", t).strip()


def _art(m: re.Match, parts: dict) -> str:
    slot = parts.get(m.group(2))
    art = _article_for(slot) if slot else m.group(1).lower()
    return f"{art.capitalize() if m.group(1)[0].isupper() else art} [{m.group(2)}]"


def _skeleton(template: str) -> str:
    return SLOT_RE.sub(" SLOT ", template)


def user_message(spec: dict) -> str:
    optional = [k for k in spec["parts"] if k not in spec["required"]]
    return _slots_line(spec["required"], optional)


def _slots_line(required: list[str], optional: list[str]) -> str:
    s = "REQUIRED: " + " ".join(f"[{k}]" for k in required)
    if optional:
        s += "\nOPTIONAL: " + " ".join(f"[{k}]" for k in optional)
    return s


def build_messages(spec: dict, seed: int = 0) -> list[dict]:
    legend = "\n".join(f"[{k}] = {_LEGEND[k]}" for k in _LEGEND if k in spec["parts"])
    msgs = [{"role": "system", "content": _SYSTEM}]
    for u, a in _examples_for(spec, seed):
        msgs += [{"role": "user", "content": f"{u}\nTEMPLATE:"}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"{user_message(spec)}\nMEANINGS:\n{legend}\nTEMPLATE:"})
    return msgs


# ---------------------------------------------------------------- gate
def gate_slotted(template: str, spec: dict, row: pd.Series, rank: int, draft: str) -> list[str]:
    """Problems with a slotted template (empty = accept). Checks the slots, the connecting words, then the filled text."""
    p: list[str] = []
    parts, required = spec["parts"], spec["required"]
    used = SLOT_RE.findall(template)
    for k in used:
        if k not in parts:
            p.append(f"unknown placeholder [{k}] (this candidate has: {', '.join('[' + x + ']' for x in parts)})")
    for k in sorted(set(used)):
        if used.count(k) > 1:
            p.append(f"placeholder [{k}] used more than once")
    for k in required:
        if k not in used:
            p.append(f"required placeholder [{k}] missing")
    if "[" in SLOT_RE.sub("", template) or "]" in SLOT_RE.sub("", template):
        p.append("stray square brackets")
    skel = _skeleton(template)
    low = skel.lower()
    if "\n" in template or re.match(r"\s*[-*\d]+[.)]?\s", template):
        p.append("list or multi-paragraph output")
    if re.search(r"\d", skel):
        p.append("digits written outside the placeholders")
    if '"' in skel or "“" in skel:
        p.append("quotation marks written outside the placeholders")
    if _FIRST_PERSON.search(skel):
        p.append("first-person pronoun")
    if len(re.findall(r"[.!?](?:\s|$)", skel)) > 3:
        p.append("more than 3 sentences")
    if re.search(r"\brank(?:ed|ing)?\b|\btop[- ]\d|\bposition\b", low):
        p.append("mentions rank or position")
    drift = sorted({m.group(0).lower() for m in _DRIFT.finditer(skel)})
    if drift:
        p.append(f"wording that changes the meaning: {drift[:3]}")
    bad_caps = [w for w in re.findall(r"(?<![.!?] )(?<!^)\b[A-Z][A-Za-z']+", skel.replace("SLOT", "slot")) if w.lower() not in _NAME_FREE_OK]
    if bad_caps:
        p.append(f"capitalised words that look like names: {bad_caps[:3]}")
    f = build_facts(row)
    name_words = {w.lower() for v in (parts.get("WHO", ""), parts.get("SKILLS", "")) for w in re.findall(r"\b[A-Z][A-Za-z0-9+.'-]+", v)} | {f.company.lower()}
    leaked = sorted(w for w in re.findall(r"[a-z][a-z'-]*", low) if w in name_words)
    if leaked:
        p.append(f"repeats a name or skill outside its placeholder: {leaked[:3]}")
    for m in SLOT_RE.finditer(template):
        before = template[max(0, m.start() - 4): m.start()].lower()
        if m.group(1) != "FIT" and re.search(r"\b(?:a|an|the) $", before):
            p.append(f"an article directly before [{m.group(1)}]: write it without 'a'/'the'")
        if m.group(1) == "FIT" and not re.search(r"(?:\ba|\ban|\(|:|-|—) ?$", template[max(0, m.start() - 4): m.start()].lower()):
            p.append("write [FIT] as 'a [FIT]' (for example 'is a [FIT]')")
    if re.search(r"\[WHO\]\s+(?:at|in|for|of|with)\b", template, re.I) or re.search(r"\b(?:at|in|for|of|with|by)\s+\[WHO\]", template, re.I):
        p.append("[WHO] already reads 'Title at Company': do not put a preposition next to it")
    has_concern = any(k.startswith("CONCERN") for k in parts)
    verdict_like = [w for w in _EVALUATIVE + _POSITIVE + _NEGATIVE if re.search(rf"\b{re.escape(w)}\b", low)]
    if not has_concern:
        verdict_like = [w for w in verdict_like if w not in ("healthy", "clean")]
    if verdict_like:
        p.append(f"adds an evaluation word of its own: {verdict_like[:3]}")
    allowed = _FUNCTION | set(re.findall(r"[a-z]+", draft.lower()))
    unknown = sorted({w for w in re.findall(r"[a-z]+", low.replace("slot", " ")) if w not in allowed and w != "s"})
    if unknown:
        p.append(f"words outside plain connectives: {unknown[:5]}")
    if re.search(r"\[YEARS\]\s+of\b(?!\s+experience\b)", template):
        p.append("[YEARS] already ends with 'of experience': do not add 'of' after it")
    for m in SLOT_RE.finditer(template):
        if m.group(1) == "QUOTE" and not re.search(r"describ|descript|state|say|said|mention|note|read|wrote|writ|summar|quot|role|profile|phrase|words", re.split(r"[.!?;]", template[: m.start()])[-1][-70:], re.I):
            p.append("introduce [QUOTE] as coming from the candidate's role description, e.g. 'a role described as [QUOTE]'")
    for m in SLOT_RE.finditer(template):  # polarity: a concern slot sits in a concern clause; a positive fact never does
        clause = SLOT_RE.sub(" SLOT ", re.split(r"[.!?;]", template[: m.start()])[-1]).lower()
        if m.group(1).startswith("CONCERN"):
            if not any(c in clause for c in _CONCERN_CUES):
                p.append(f"[{m.group(1)}] is not framed as a concern")
        elif m.group(1) in _POSITIVE_SLOTS and any(re.search(rf"\b{re.escape(w)}", clause) for w in _CONCERN_WORDS):
            p.append(f"[{m.group(1)}] is a positive fact but sits in a concern sentence")
    if not has_concern and any(re.search(rf"\b{w}\b", low) for w in _CONCERN_WORDS):
        p.append("concern wording on a candidate with no concern")
    if not p:
        text = fill_slots(template, spec)
        n_words = len(text.split())
        if not 25 <= n_words <= 100:
            p.append(f"length {n_words} words")
        if not _relation_ok(text, f.title, f.company):
            p.append("who works where changed: [WHO] must be the subject phrase, e.g. 'Title at Company'")
        words5 = re.findall(r"\d+(?:\.\d+)?|[a-z'-]+", text.lower())
        grams = [" ".join(words5[i:i + 5]) for i in range(len(words5) - 4)]
        if len(grams) != len(set(grams)):
            p.append("repeats itself")
        sk_problems, _ = _verify_skill_triples(text, row)
        p += sk_problems
        p += audit_reasoning(text, row, skip_skill_parse=True)
    return sorted(set(p))


# ---------------------------------------------------------------- generation (offline) and cache use (ranking time)
MAX_SAME_SKELETON = 2  # a skeleton (template words + slot order) used by more rows than this is replaced when a differing one exists


def _skeleton_key(template: str) -> str:
    return re.sub(r"\s+", " ", template).strip().lower()


def _pick_diverse(cands: list[list[tuple[int, str]]]) -> list[tuple[int, str] | None]:
    """For each row (in order) pick the accepted template whose skeleton has been used least so far."""
    usage: dict[str, int] = {}
    chosen: list = []
    for c in cands:
        if not c:
            chosen.append(None)
            continue
        best = min(c, key=lambda x: usage.get(_skeleton_key(x[1]), 0))
        usage[_skeleton_key(best[1])] = usage.get(_skeleton_key(best[1]), 0) + 1
        chosen.append(best)
    return chosen


def generate_reasonings_slotted(candidates_df: pd.DataFrame, diagnostics: list | None = None, generator=None, model_name: str = DEFAULT_MODEL,
                                attempts: int = 4, cache_path=None) -> list[str]:
    """Slot-filled template from the LLM -> gate -> fill, else the grounded draft.
    Attempt 1 is greedy; later attempts are sampled and only cover rows that still have no accepted template or whose accepted
    template is a skeleton many other rows share (the example order and the forced opener rotate per attempt). Every accepted
    template is kept, and the least-reused skeleton wins. `generator(list_of_message_lists) -> list_of_str` is injectable for tests."""
    rows = [row for _, row in candidates_df.iterrows()]
    ranks = [int(r["rank"]) if "rank" in candidates_df.columns and pd.notna(r.get("rank")) else i + 1 for i, r in enumerate(rows)]
    facts = [build_facts(r) for r in rows]
    drafts = [compose_reasoning(f, k) for f, k in zip(facts, ranks)]
    specs = [build_slot_parts(f, k) for f, k in zip(facts, ranks)]
    state = {"sample": False}
    generator = generator or (lambda msgs: _generate(msgs, model_name, max_new_tokens=90, sample=state["sample"]))
    n = len(rows)
    cands: list[list[tuple[int, str]]] = [[] for _ in range(n)]
    last, last_problems = [""] * n, [[] for _ in range(n)]
    for attempt in range(1, attempts + 1):
        chosen = _pick_diverse(cands)
        usage: dict[str, int] = {}
        for c in chosen:
            if c:
                usage[_skeleton_key(c[1])] = usage.get(_skeleton_key(c[1]), 0) + 1
        todo = [i for i in range(n) if chosen[i] is None or usage[_skeleton_key(chosen[i][1])] > MAX_SAME_SKELETON]
        if not todo:
            break
        msgs = []
        for i in todo:
            m = build_messages(specs[i], _seed_of(rows[i]) + attempt - 1)
            if not cands[i] and last[i]:  # no accepted template yet: show the model what was wrong with its last try
                m += [{"role": "assistant", "content": last[i]},
                      {"role": "user", "content": "That is not acceptable: " + "; ".join(last_problems[i][:4]) +
                       ". Write the TEMPLATE again, fixing exactly these problems.\n" + user_message(specs[i]) + "\nTEMPLATE:"}]
            m.append({"role": "assistant", "content": _OPENERS[(_seed_of(rows[i]) + attempt - 1) % len(_OPENERS)], "prefill": True})
            msgs.append(m)
        try:
            state["sample"] = attempt > 1
            outs = generator(msgs)
        except Exception as e:
            print(f"LLM unavailable ({e}); using the grounded drafts.")
            break
        for i, raw in zip(todo, outs):
            text = _clean(raw)
            probs = gate_slotted(text, specs[i], rows[i], ranks[i], drafts[i]) if text else ["empty output"]
            last[i], last_problems[i] = text, probs
            if not probs and all(text != c[1] for c in cands[i]):
                cands[i].append((attempt, text))
        print(f"  attempt {attempt}: {len(todo)} rows tried, {sum(bool(c) for c in cands)}/{n} rows now have an accepted template")
    chosen = _pick_diverse(cands)
    final = [fill_slots(c[1], specs[i]) if c else drafts[i] for i, c in enumerate(chosen)]
    if diagnostics is not None:
        for i in range(n):
            diagnostics.append({"candidate_id": rows[i]["candidate_id"], "rank": ranks[i], "accepted": chosen[i] is not None, "attempt": chosen[i][0] if chosen[i] else 0,
                                "n_candidates": len(cands[i]), "problems": [] if chosen[i] else last_problems[i], "draft": drafts[i],
                                "template": chosen[i][1] if chosen[i] else last[i], "llm": final[i] if chosen[i] else ""})
    if cache_path is not None:
        cache = load_cache(cache_path)
        for i in range(n):
            if chosen[i]:
                cache[draft_key(drafts[i])] = {"template": chosen[i][1], "rewrite": final[i], "model": model_name, "attempt": chosen[i][0], "mode": "slotted"}
        save_cache(cache_path, cache)
    ok = sum(c is not None for c in chosen)
    print(f"Slotted LLM text accepted for {ok}/{n} rows ({len({_skeleton_key(c[1]) for c in chosen if c})} distinct skeletons); the rest use the grounded draft.")
    return final


def generate_reasonings_slotted_cached(candidates_df: pd.DataFrame, cache_path, diagnostics: list | None = None) -> list[str]:
    """Use cached templates. Each is re-gated and RE-FILLED from the current record and rank; a miss or failed check falls back to the grounded draft."""
    cache = load_cache(cache_path)
    out, hits, used = [], 0, 0
    for i, (_, r) in enumerate(candidates_df.iterrows()):
        k = int(r["rank"]) if "rank" in candidates_df.columns and pd.notna(r.get("rank")) else i + 1
        f = build_facts(r)
        draft, spec = compose_reasoning(f, k), build_slot_parts(f, k)
        entry = cache.get(draft_key(draft))
        problems = ["cache miss"] if not entry or "template" not in entry else gate_slotted(entry["template"], spec, r, k, draft)
        hits += bool(entry)
        ok = entry is not None and not problems
        used += ok
        out.append(fill_slots(entry["template"], spec) if ok else draft)
        if diagnostics is not None:
            diagnostics.append({"candidate_id": r["candidate_id"], "rank": k, "accepted": ok, "problems": problems, "draft": draft, "llm": out[-1] if ok else ""})
    print(f"Cached slotted rewrites: {used}/{len(out)} rows used ({hits} cache hits); the rest use the grounded draft.")
    return out
