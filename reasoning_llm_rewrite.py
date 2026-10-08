"""
@file Optional LLM polish on top of the fact-grounded reasoning (`--reasoning-mode grounded-llm`).
@package online_ranking

The LLM is never the source of facts:
  1. reasoning_grounded.py writes a DRAFT from the candidate's own record (already audited);
  2. a local instruction model rewrites the draft into more natural prose. The prompt gives an explicit
     KEEP list for the row (fit wording, title, company, skills, quote, concern and availability facts),
     states that the candidate is always the subject, and shows worked examples;
  3. `gate_rewrite` accepts the rewrite only if it keeps those facts, introduces no new number, name or
     quality adjective, changes nobody's employer, shows no job-posting drift, contradicts no tone rule,
     and passes the same record audit as the draft;
  4. a row that fails gets ONE repair attempt: the gate's exact complaints are fed back to the model;
  5. a row that still fails falls back to the draft, so the worst case equals the plain grounded output.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from reasoning_grounded import _VERDICTS, _tier, audit_reasoning, build_facts, compose_reasoning

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_VERDICT_WORDS = sorted({v.lower() for vs in _VERDICTS.values() for v in vs}, key=len, reverse=True)
_POSITIVE = ("strong fit", "clear fit", "very good", "excellent", "outstanding", "exceptional", "top-tier", "ideal", "perfect")
_NEGATIVE = ("borderline", "marginal", "partial", "weak fit", "poor fit")
_RETAIN = ("notice", "open-to-work", "inactiv", "response rate", "github", "no product-company", "evaluation", "relocat",
           "below the jd", "above the jd", "employers", "retrieval or vector-search", "relative score")
_EVALUATIVE = ("high", "low", "strong", "great", "excellent", "impressive", "outstanding", "solid", "good", "poor", "weak", "clean", "healthy")
_ALLOWED_CAPS = {"jd", "ndcg", "mrr", "github", "pune", "noida", "india", "nlp", "ml", "ai", "llm", "rag"}
_FIRST_PERSON = re.compile(r"\b(i|my|we|our|me|us)\b", re.I)
# wording that turns candidate facts into a job posting, or upgrades a claim beyond the draft
_DRIFT = re.compile(
    r"\b(recruit(?:ing|ment|s|ed)|hiring|hires?|vacanc\w*|applicants?|positions?|offers?|offering|seeking|required|requires|requirement"
    r"|requiring|expertise|proven|highly|excellent|seasoned|extensive|impressive|exceptional|renowned|track record)\b"
    r"|\brole (?:is )?(?:requiring|requires|located|based|description provided)\b|\bfor (?:the|this) (?:role|position)\b"
    r"|\bholding back\b|\bsolid matches\b|\bthere is an? (?:solid|good|moderate|strong) match\b|\b(?:and|with|along with) (?-i:[A-Z][a-z]+) inside\b", re.I)

_SYSTEM = (
    "You edit short recruiter notes about ONE job candidate. The candidate is always the subject of the note. Rewrite the DRAFT so it reads "
    "naturally, using different sentence structure and connecting words, while keeping every fact.\n"
    "Rules:\n"
    "- The candidate is the subject. Never say a company offers, recruits or hires, and never describe a position or role being offered.\n"
    "- Copy exactly, character for character: the job title, the company, every skill written as Name (level, N mo), every number, and any text in double quotes.\n"
    "- Keep every concern and availability fact from the draft, written as facts about the candidate.\n"
    "- Keep the fit wording given in KEEP (for example strong fit or borderline fit).\n"
    "- Do not add skills, companies, numbers or claims, and do not add praise words such as expert, proven, highly or excellent unless they are in the draft.\n"
    "- Write 2 sentences, 35 to 65 words. Never mention a rank."
)
_EXAMPLES = [
    ("Strong fit: Senior ML Engineer at Acme Labs with 7.1 years of experience, showing JD-relevant skills including FAISS (expert, 60 mo) and "
     "Python (expert, 84 mo) and product-company experience. The gap to watch: a 90-day notice period against the JD's preference for under 30.",
     "strong fit | Senior ML Engineer | Acme Labs | 7.1 years | FAISS (expert, 60 mo) | Python (expert, 84 mo) | product-company experience | "
     "a 90-day notice period against the JD's preference for under 30",
     "Senior ML Engineer at Acme Labs, with 7.1 years of experience, is a strong fit: the profile lists FAISS (expert, 60 mo) and Python (expert, 84 mo) "
     "and includes product-company experience. The gap to watch is a 90-day notice period against the JD's preference for under 30."),
    ("Borderline fit - Nova's Search Engineer, 4.9 years in, with hands-on skills including Pinecone (advanced, 56 mo) and Semantic Search (advanced, 23 mo) "
     "and a role described as \"Built recommendation-style features at a mid-stage startup\". Held back by 4.9 years of experience, below the JD's 5-9 year band; no open-to-work flag.",
     "borderline fit | Search Engineer | Nova | 4.9 years | Pinecone (advanced, 56 mo) | Semantic Search (advanced, 23 mo) | "
     "\"Built recommendation-style features at a mid-stage startup\" | 4.9 years of experience, below the JD's 5-9 year band | no open-to-work flag",
     "Nova's Search Engineer has 4.9 years of experience and lists Pinecone (advanced, 56 mo) and Semantic Search (advanced, 23 mo), plus a role described as "
     "\"Built recommendation-style features at a mid-stage startup\". It is a borderline fit: 4.9 years of experience is below the JD's 5-9 year band and there is no open-to-work flag."),
    ("Good fit - Kite's Applied Scientist, 6.2 years in, with JD-relevant skills including Weaviate (expert, 70 mo) and BM25 (advanced, 33 mo). "
     "Signals are healthy: 71% recruiter response rate, 30-day notice, based in Pune.",
     "good fit | Applied Scientist | Kite | 6.2 years | Weaviate (expert, 70 mo) | BM25 (advanced, 33 mo) | 71% recruiter response rate | 30-day notice | based in Pune",
     "At Kite, a 6.2-year Applied Scientist is a good fit, listing Weaviate (expert, 70 mo) and BM25 (advanced, 33 mo). The candidate has a 71% recruiter response rate, a 30-day notice and is based in Pune."),
]


# ---------------------------------------------------------------- helpers
def _unquoted(t: str) -> str:
    return re.sub(r'"[^"]*"', " ", t)


def _skill_claims(t: str) -> list[str]:
    out = []
    u = _unquoted(t)
    for m in re.finditer(r"\((beginner|intermediate|advanced|expert), (\d+) mo\)", u):
        before = u[: m.start()].rstrip()
        sep = max(((before.rfind(sp), len(sp)) for sp in ("including ", ", ", " and ")), key=lambda x: x[0])
        name = before[sep[0] + sep[1]:].strip() if sep[0] >= 0 else before
        out.append(f"{name} ({m.group(1)}, {m.group(2)} mo)")
    return out


def _numbers(t: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", _unquoted(t)))


def _caps(t: str) -> set[str]:
    words = set()
    for sent in re.split(r"(?<=[.!?])\s+", _unquoted(t)):
        for w in re.findall(r"\b[A-Z][A-Za-z0-9+.'/-]*\b", sent)[1:]:
            lw = re.sub(r"['’]s$", "", w.lower().rstrip(".'"))
            words.update(x for x in re.split(r"[/-]", lw) if x)
    return words


def _second_sentence_facts(draft: str) -> list[str]:
    """The concern / availability facts of the draft's last sentence, split into short phrases."""
    last = re.split(r"(?<=[.!?])\s+(?=[A-Z])", draft.strip())[-1].rstrip(".")
    last = re.sub(r"^(Main concern|The gap to watch|Caveat|Held back by|Signals are healthy|Availability is clean|Availability checks out|"
                  r"No red flags in availability|Nothing disqualifying on record[^.]*?\. ?Availability|No single disqualifying gap, but a lower relative score than "
                  r"higher-ranked profiles; availability)\s*:?\s*", "", last, flags=re.I).strip("() ")
    return [p.strip() for p in re.split(r";|, (?=\d|immediate|based)", last) if p.strip()]


def keep_list(draft: str, row: pd.Series) -> str:
    """Explicit per-row checklist of what the rewrite must keep verbatim."""
    f = build_facts(row)
    verdict = next((v for v in _VERDICT_WORDS if v in draft.lower()), "")
    parts = [verdict, f.title, f.company, f"{f.yoe:g} years"] + _skill_claims(draft) + [f'"{q}"' for q in re.findall(r'"([^"]+)"', draft)] + _second_sentence_facts(draft)
    return " | ".join(p for p in parts if p)


def prefill_phrase(row: pd.Series, rank: int) -> str:
    """The opening of the rewrite, fixed so the candidate is always the subject and the employer can never be swapped."""
    f = build_facts(row)
    seed = int(re.sub(r"\D", "", f.candidate_id) or 0)
    yrs = f"{f.yoe:g}"
    return (f"{f.title} at {f.company}", f"{f.company}'s {f.title}", f"At {f.company}, a {yrs}-year {f.title}", f"{yrs}-year {f.title} at {f.company}")[seed % 4]


def _quote_rule(draft: str) -> str:
    qs = re.findall(r'"([^"]+)"', draft)
    return f'QUOTE RULE: your answer must contain this text exactly as written, with its double quotes: "{qs[0]}"\n' if qs else ""


def build_messages(draft: str, row: pd.Series) -> list[dict]:
    msgs = [{"role": "system", "content": _SYSTEM}]
    for d, keep, r in _EXAMPLES:
        msgs += [{"role": "user", "content": f"DRAFT: {d}\nKEEP: {keep}\n{_quote_rule(d)}REWRITE:"}, {"role": "assistant", "content": r}]
    msgs.append({"role": "user", "content": f"DRAFT: {draft}\nKEEP: {keep_list(draft, row)}\n{_quote_rule(draft)}REWRITE:"})
    return msgs


def _verify_skill_triples(text: str, row: pd.Series) -> tuple[list[str], list[str]]:
    """Every "(level, N mo)" in `text` must belong to a skill the candidate really lists. Returns (problems, verified claims)."""
    try:
        owned = {s["name"]: s for s in json.loads(row["skills_json"])}
    except (TypeError, KeyError, ValueError):
        owned = {}
    scan, verified, problems = _unquoted(text), [], []
    for name in sorted(owned, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(name)} \((beginner|intermediate|advanced|expert), (\d+) mo\)", scan):
            s = owned[name]
            if s.get("proficiency") == m.group(1) and int(s.get("duration_months") or 0) == int(m.group(2)):
                verified.append(f"{name} ({m.group(1)}, {m.group(2)} mo)")
            else:
                problems.append(f"skill detail wrong: {name}")
        scan = re.sub(rf"(?<![\w-]){re.escape(name)} \((?:beginner|intermediate|advanced|expert), \d+ mo\)", " ", scan)
    for _ in re.finditer(r"\((?:beginner|intermediate|advanced|expert), \d+ mo\)", scan):
        problems.append("a skill claim does not match any skill in the profile")
    return problems, verified


def _relation_ok(text: str, title: str, company: str) -> bool:
    t, c = re.escape(title), re.escape(company)
    return any(re.search(p, text, re.I) for p in (
        rf"{t} (?:at|for|with) {c}", rf"{c}['’]s (?:[\d.]+-year )?{t}", rf"at {c}, an? (?:[\d.]+-year )?{t}", rf"{t}, (?:working )?at {c}"))


def gate_rewrite(rewrite: str, draft: str, row: pd.Series, rank: int) -> list[str]:
    """Problems that disqualify `rewrite` (empty list = accept). Optional positives (product-company, the 5-9 band, the quote,
    a third skill) may be dropped; concerns, availability facts, the verdict, the title/company relation and two skills may not."""
    p: list[str] = []
    low = rewrite.lower()
    n_words = len(rewrite.split())
    if not 25 <= n_words <= 95:
        p.append(f"length {n_words} words (need 35 to 65)")
    if len(re.findall(r"[.!?](?:\s|$)", rewrite)) > 3:
        p.append("more than 3 sentences")
    if "\n" in rewrite or re.match(r"\s*[-*\d]+[.)]?\s", rewrite):
        p.append("list or multi-paragraph output")
    if re.search(r"#\s?\d|\branked\b|\brank\s*(?:#|no\.?|number|\d)|\bposition\s*(?:#|\d)|\b(?:top|bottom)[- ]\d+\b|\b(?:\d+(?:st|nd|rd|th))\s+(?:place|rank|position)\b",
                 _unquoted(rewrite).lower()):
        p.append("mentions rank or position")
    if _FIRST_PERSON.search(_unquoted(rewrite)):
        p.append("first-person pronoun")
    drift_new = sorted({m.group(0).lower() for m in _DRIFT.finditer(_unquoted(rewrite))} - {m.group(0).lower() for m in _DRIFT.finditer(_unquoted(draft))})
    if drift_new:
        p.append(f"wording not in the draft that changes the meaning: {drift_new[:3]}")
    dl, ul = draft.lower(), _unquoted(rewrite).lower()
    for kw in _RETAIN:  # a concern or fact the draft never made
        if kw in ul and kw not in _unquoted(dl):
            p.append(f"claims something the draft does not say: {kw}")
    added_eval = [w for w in _EVALUATIVE if re.search(rf"\b{w}\b", ul) and not re.search(rf"\b{w}\b", _unquoted(dl))]
    if added_eval:
        p.append(f"adds an evaluation not in the draft: {added_eval[:3]}")
    # grammar slips the model makes: odd mid-sentence capitals ("listed Skills include") and "a 4.8 years of experience"
    odd_caps = [w for w in re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]{2,}", _unquoted(rewrite))
                if w.lower() in _unquoted(dl) and w not in _unquoted(draft)]
    if odd_caps:
        p.append(f"odd capitalisation: {odd_caps[:2]}")
    if re.search(r"\b(?:a|an) \d+(?:\.\d+)? years\b|\bhas an? \d+(?:\.\d+)?-day\b|\b(?:a|an) \d+(?:\.\d+)?-year experience\b", ul):
        p.append("grammar: 'a N years'")
    words5 = re.findall(r"\d+(?:\.\d+)?|[a-z'-]+", ul)  # numbers stay whole, so "4.9 years of experience" is one phrase
    grams = [" ".join(words5[i:i + 5]) for i in range(len(words5) - 4)]
    if len(grams) != len(set(grams)):
        p.append("repeats itself")
    f = build_facts(row)
    if f.title not in rewrite or f.company not in rewrite:
        p.append("title or company changed")
    elif not _relation_ok(rewrite, f.title, f.company):
        p.append("who works where changed: the candidate must be the subject, e.g. 'Title at Company'")
    skill_problems, kept = _verify_skill_triples(rewrite, row)
    p += skill_problems
    draft_claims = _skill_claims(draft)
    for c in draft_claims:
        name = c.split(" (")[0]
        if name.lower() in ul and c not in rewrite:
            p.append(f"skill '{name}' named without its exact level and months")
    if len([c for c in draft_claims if c in kept]) < min(2, len(draft_claims)):
        p.append("fewer than two of the draft's skills kept exactly as Name (level, N mo)")
    draft_quotes = re.findall(r'"([^"]+)"', draft)
    if draft_quotes and not any(f'"{q}"' in rewrite for q in draft_quotes):
        p.append("the verbatim quote was dropped: keep it exactly, with its double quotes")
    for q in re.findall(r'"([^"]+)"', rewrite):
        if q not in draft_quotes:
            p.append("a quote was altered or invented")
    dn, rn = _numbers(draft), _numbers(rewrite)
    optional = re.sub(r"experience inside the JD's 5-9 year band", " ", draft)
    required = _numbers(re.sub(r"\((?:beginner|intermediate|advanced|expert), \d+ mo\)", " ", optional))
    if required - rn:
        p.append(f"numbers dropped: {sorted(required - rn)}")
    if rn - dn:
        p.append(f"new numbers not in the draft: {sorted(rn - dn)}")
    for kw in _RETAIN:
        if kw in draft.lower() and kw not in low:
            p.append(f"lost: {kw}")
    verdict = next((v for v in _VERDICT_WORDS if v in draft.lower()), None)
    if verdict and verdict not in low:
        p.append(f"verdict wording lost: keep '{verdict}'")
    new_caps = _caps(rewrite) - _caps(draft) - _ALLOWED_CAPS
    new_caps = {c for c in new_caps if c not in draft.lower()}
    if new_caps:
        p.append(f"new capitalised terms: {sorted(new_caps)[:4]}")
    tier = _tier(rank)
    if tier == 1 and any(w in low for w in _NEGATIVE):
        p.append("weak-fit wording on a top-10 row")
    if tier == 4 and any(w in low for w in _POSITIVE):
        p.append("strong-fit wording on a rank>70 row")
    p += audit_reasoning(rewrite, row, skip_skill_parse=True)  # remaining record checks (years, quotes, response %, notice, title/company)
    return sorted(set(p))


# ---------------------------------------------------------------- generation
def _clean(raw: str) -> str:
    t = raw.strip().split("\n\n")[0].replace("\n", " ").strip()
    t = re.sub(r"^\s*(REWRITE|Rewrite|Summary)\s*:\s*", "", t)
    if len(t) > 1 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
        t = t[1:-1].strip()
    if t and not t.endswith((".", "!", "?")):
        m = list(re.finditer(r"[.!?](?:\s|$)", t))
        t = t[: m[-1].start() + 1] if m else t
    return re.sub(r"\s+", " ", t)


# ------------------------------------------------------------ offline cache (the LLM runs once, offline; ranking only reads it)
def draft_key(draft: str) -> str:
    import hashlib
    return hashlib.sha1(draft.encode("utf-8")).hexdigest()


def load_cache(path) -> dict:
    from pathlib import Path
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_cache(path, cache: dict) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def generate_reasonings_cached(candidates_df: pd.DataFrame, cache_path, diagnostics: list | None = None) -> list[str]:
    """Use precomputed rewrites. Each cached rewrite is looked up by the exact grounded draft and RE-CHECKED by the gate against the
    candidate's record and current rank; a miss or a failed check falls back to the draft. No model is loaded."""
    cache = load_cache(cache_path)
    rows = [row for _, row in candidates_df.iterrows()]
    ranks = [int(r["rank"]) if "rank" in candidates_df.columns and pd.notna(r.get("rank")) else i + 1 for i, r in enumerate(rows)]
    out, hits, used = [], 0, 0
    for r, k in zip(rows, ranks):
        draft = compose_reasoning(build_facts(r), k)
        entry = cache.get(draft_key(draft))
        problems = ["cache miss"] if not entry else gate_rewrite(entry["rewrite"], draft, r, k)
        hits += bool(entry)
        ok = entry is not None and not problems
        used += ok
        out.append(entry["rewrite"] if ok else draft)
        if diagnostics is not None:
            diagnostics.append({"candidate_id": r["candidate_id"], "rank": k, "accepted": ok, "problems": problems, "draft": draft, "llm": entry["rewrite"] if entry else ""})
    print(f"Cached LLM rewrites: {used}/{len(out)} rows used ({hits} cache hits); the rest use the grounded draft.")
    return out


_MODELS: dict = {}


def _load(model_name: str):
    if model_name not in _MODELS:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: loads from the local Hugging Face cache
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        import os
        import torch
        kwargs = {"device_map": "cpu"}
        if os.environ.get("LLM_DTYPE"):  # e.g. LLM_DTYPE=bfloat16 halves the memory of a 1.5B+ model (native on avx512_bf16 CPUs)
            kwargs["dtype"] = getattr(torch, os.environ["LLM_DTYPE"])
        _MODELS[model_name] = (AutoModelForCausalLM.from_pretrained(model_name, **kwargs), tok)
    return _MODELS[model_name]


def _generate(message_lists: list[list[dict]], model_name: str = DEFAULT_MODEL, batch_size: int = 25, max_new_tokens: int = 100, sample: bool = False) -> list[str]:
    import torch

    model, tok = _load(model_name)
    outs: list[str] = []
    for i in range(0, len(message_lists), batch_size):
        chunk = message_lists[i:i + batch_size]
        # a trailing {"role": "assistant", "prefill": True} message is the forced start of the answer
        prefills = [m[-1]["content"] if m and m[-1].get("prefill") else "" for m in chunk]
        prompts = [tok.apply_chat_template([x for x in m if not x.get("prefill")], tokenize=False, add_generation_prompt=True) + pf for m, pf in zip(chunk, prefills)]
        inputs = tok(prompts, padding=True, return_tensors="pt").to("cpu")
        torch.manual_seed(0)
        opts = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else {"do_sample": False}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **opts)
        for j in range(len(prompts)):
            outs.append(prefills[j] + tok.decode(gen[j][inputs.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def generate_reasonings_llm_rewrite(candidates_df: pd.DataFrame, diagnostics: list | None = None, generator=None,
                                    model_name: str = DEFAULT_MODEL, attempts: int = 2, use_prefill: bool = True, cache_path=None) -> list[str]:
    """Grounded draft -> LLM rewrite -> gate -> (one repair attempt) -> accept, else fall back to the draft.
    `generator(list_of_message_lists) -> list_of_str` is injectable for tests."""
    rows = [row for _, row in candidates_df.iterrows()]
    ranks = [int(r["rank"]) if "rank" in candidates_df.columns and pd.notna(r.get("rank")) else i + 1 for i, r in enumerate(rows)]
    drafts = [compose_reasoning(build_facts(r), k) for r, k in zip(rows, ranks)]
    generator = generator or (lambda msgs: _generate(msgs, model_name))
    n = len(rows)
    final, accepted_on, last_text, last_problems = list(drafts), [0] * n, [""] * n, [[] for _ in range(n)]
    messages = [build_messages(d, r) for d, r in zip(drafts, rows)]
    prefills = [prefill_phrase(r, k) for r, k in zip(rows, ranks)]
    pending = list(range(n))
    for attempt in range(1, attempts + 1):
        if not pending:
            break
        try:
            outs = generator([messages[i] + ([{"role": "assistant", "content": prefills[i], "prefill": True}] if use_prefill else []) for i in pending])
        except Exception as e:  # model missing/offline: the draft is always a valid answer
            print(f"LLM rewrite unavailable ({e}); using the grounded drafts.")
            break
        still = []
        for i, raw in zip(pending, outs):
            text = _clean(raw)
            probs = gate_rewrite(text, drafts[i], rows[i], ranks[i]) if text else ["empty output"]
            last_text[i], last_problems[i] = text, probs
            if not probs:
                final[i], accepted_on[i] = text, attempt
            else:
                still.append(i)
                messages[i] = messages[i] + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "That rewrite is not acceptable: " + "; ".join(probs[:4]) +
                     ". Rewrite the DRAFT again, fixing exactly these problems and following all the rules.\nKEEP: " + keep_list(drafts[i], rows[i]) + "\nREWRITE:"}]
        pending = still
    if diagnostics is not None:
        for i in range(n):
            diagnostics.append({"candidate_id": rows[i]["candidate_id"], "rank": ranks[i], "accepted": bool(accepted_on[i]), "attempt": accepted_on[i],
                                "problems": [] if accepted_on[i] else last_problems[i], "draft": drafts[i], "llm": last_text[i]})
    if cache_path is not None:  # offline build: remember every accepted rewrite (keyed by its exact draft)
        cache = load_cache(cache_path)
        for i in range(n):
            if accepted_on[i]:
                cache[draft_key(drafts[i])] = {"rewrite": final[i], "model": model_name, "attempt": accepted_on[i]}
        save_cache(cache_path, cache)
    a1, a2 = sum(x == 1 for x in accepted_on), sum(x == 2 for x in accepted_on)
    print(f"LLM rewrite accepted for {a1 + a2}/{n} rows ({a1} on the first attempt, {a2} after repair); the rest use the grounded draft.")
    return final
