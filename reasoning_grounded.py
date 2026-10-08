"""
@file Deterministic, fact-grounded recruiter reasoning for the final top-100.
@package online_ranking

Every claim in a generated reasoning is read from the candidate's own record:
title, company, years of experience, skills that appear in the candidate's skill
list (with proficiency and months used), short VERBATIM quotes from their role
descriptions, and platform signals (response rate, notice period, last active,
location). Nothing is inferred by a language model, so nothing can be
hallucinated, and `audit_reasoning` re-checks each claim against the record.

Tone follows the rank tier (strong / good / reasonable / borderline) and the
concerns are the real ones for that candidate, so the text stays consistent with
the rank. Different candidates surface different facts and use different frames.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import pandas as pd

from features_utils import AS_OF_DATE

# JD skill groups -> weight. Only skills that actually appear in the candidate's list are ever named.
_GROUPS: dict[str, tuple[int, tuple[str, ...]]] = {
    "search": (3, ("Semantic Search", "Vector Search", "Information Retrieval", "Information Retrieval Systems",
                   "Search Infrastructure", "BM25", "Indexing Algorithms")),
    "embeddings": (3, ("Embeddings", "Sentence Transformers", "Text Encoders")),
    "vector_infra": (3, ("FAISS", "Qdrant", "Weaviate", "Milvus", "Pinecone", "pgvector", "Elasticsearch",
                         "OpenSearch", "Haystack")),
    "ranking": (3, ("Learning to Rank", "Recommendation Systems")),
    "llm_ft": (1, ("LoRA", "QLoRA", "PEFT", "Fine-tuning LLMs", "Model Adaptation")),
    "python": (1, ("Python",)),
}
_SKILL_GROUP = {name: g for g, (_, names) in _GROUPS.items() for name in names}
_GROUP_WEIGHT = {g: w for g, (w, _) in _GROUPS.items()}
_CORE_GROUPS = {g for g, w in _GROUP_WEIGHT.items() if w >= 3}
_PROF_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
MIN_SKILL_MONTHS = 12  # ignore skills claimed with almost no usage time

_QUOTE_KEYWORDS = re.compile(
    r"retriev|ranking|\brank|search|recommend|embedding|vector|relevance|evaluat|ndcg|a/b|\bindex", re.I)
_EVAL_WORDS = re.compile(r"evaluat|ndcg|\bmrr\b|a/b|offline metric|online metric", re.I)
_FIRST_PERSON = re.compile(r"\b(i|my|we|our|me|us)\b", re.I)

# Verdict wording by rank tier; picked by a hash of the candidate id.
_VERDICTS = {
    1: ("Strong fit", "Clear fit", "Very good match"),
    2: ("Good fit", "Solid match", "Good match"),
    3: ("Reasonable fit", "Partial-to-good match", "Moderate match"),
    4: ("Borderline fit", "Partial match", "Marginal match"),
}


def _tier(rank: int) -> int:
    return 1 if rank <= 10 else 2 if rank <= 35 else 3 if rank <= 70 else 4


def _seed(candidate_id: str) -> int:
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16)


def _load(row, key):
    try:
        return json.loads(row[key])
    except (TypeError, KeyError, json.JSONDecodeError):
        return []


def _num(x, default=None):
    try:
        return default if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return default


def _yrs(y: float) -> str:
    return f"{y:g}"


# --------------------------------------------------------------------------- facts
@dataclass
class Facts:
    candidate_id: str
    title: str
    company: str
    yoe: float
    skills_named: list[dict] = field(default_factory=list)   # {name, proficiency, months}
    has_core_skill: bool = False
    quote: str | None = None
    product_ok: bool = False
    resp: float | None = None
    last_active: str | None = None
    idle_days: int | None = None
    notice: int | None = None
    open_to_work: bool | None = None
    github: float | None = None
    country: str = ""
    city: str = ""
    relocate: bool | None = None
    n_employers: int = 0
    avg_tenure_months: float | None = None
    eval_mentioned: bool = True
    descriptions: list[str] = field(default_factory=list)


def _select_skills(skills: list[dict], k: int = 3) -> tuple[list[dict], bool]:
    cands = []
    for s in skills:
        g = _SKILL_GROUP.get(s.get("name"))
        months = int(s.get("duration_months") or 0)
        prof = s.get("proficiency", "")
        if not g or months < MIN_SKILL_MONTHS or prof == "beginner":
            continue
        cands.append((_GROUP_WEIGHT[g], _PROF_RANK.get(prof, 0), months, g, s["name"], prof))
    cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    chosen, used = [], set()
    for c in cands:  # first pass: one skill per group, so three vector databases don't crowd out ranking skills
        if c[3] not in used:
            chosen.append(c)
            used.add(c[3])
        if len(chosen) == k:
            break
    for c in cands:
        if len(chosen) == k:
            break
        if c not in chosen:
            chosen.append(c)
    has_core = any(c[3] in _CORE_GROUPS for c in cands)
    return [{"name": c[4], "proficiency": c[5], "months": c[2]} for c in chosen], has_core


def _pick_quote(roles: list[dict], max_words: int = 16) -> str | None:
    """Best JD-relevant sentence from the role descriptions (current role first), verbatim, truncated on a word boundary."""
    scored = []
    for order, role in enumerate(roles):
        for sent in re.split(r"(?<=[.!?])\s+", (role.get("description") or "").strip()):
            hits = len(_QUOTE_KEYWORDS.findall(sent))
            n_words = len(sent.split())
            # more keyword hits wins; a complete short sentence beats one that must be cut;
            # ties go to the more recent role
            if hits:
                scored.append((hits * 10 - order - (0 if n_words <= max_words else 6), sent.strip()))
    scored.sort(key=lambda x: -x[0])
    for _, sent in scored:
        quote = _trim_quote(sent, max_words)
        if quote and _QUOTE_KEYWORDS.search(quote):  # the KEPT text must still show why it was chosen
            return quote
    return None


def _trim_quote(sent: str, max_words: int) -> str | None:
    words = sent.rstrip(".!? ").split()
    if len(words) <= max_words:
        return " ".join(words)
    # too long: cut at the last clause boundary inside the word limit if that keeps a meaningful chunk
    head = " ".join(words[:max_words])
    cut = max(head.rfind(","), head.rfind(";"), head.rfind(" — "), head.rfind(" - "))
    out = head[:cut].rstrip(",;:—- ") if cut > len(head) * 0.5 else head.rstrip(",;:—- ") + "…"
    if out.count("(") > out.count(")"):  # never leave a parenthesis open
        out = out[: out.rfind("(")].rstrip(",;:—- ")
    return out or None


def build_facts(row: pd.Series) -> Facts:
    roles = _load(row, "career_history_json")
    roles_cur_first = sorted(roles, key=lambda r: (not r.get("is_current", False)))
    skills, has_core = _select_skills(_load(row, "skills_json"))
    active = row.get("redrob_last_active_date")
    try:
        idle = (AS_OF_DATE - pd.to_datetime(active).date()).days
    except (TypeError, ValueError):
        idle = None
    gh = _num(row.get("redrob_github_activity_score"))
    notice = _num(row.get("redrob_notice_period_days"))
    loc = str(row.get("location", ""))
    otw = row.get("redrob_open_to_work_flag")
    reloc = row.get("redrob_willing_to_relocate")
    descs = [r.get("description") or "" for r in roles]
    narrative = str(row.get("narrative_text", ""))
    return Facts(
        candidate_id=str(row["candidate_id"]),
        title=str(row.get("current_title", "")).strip(),
        company=str(row.get("current_company", "")).strip(),
        yoe=float(_num(row.get("years_of_experience"), 0.0)),
        skills_named=skills,
        has_core_skill=has_core,
        quote=_pick_quote(roles_cur_first),
        product_ok=bool(row.get("has_product_company_experience")) and not bool(row.get("is_consulting_only")),
        resp=_num(row.get("redrob_recruiter_response_rate")),
        last_active=str(active) if active is not None and not pd.isna(active) else None,
        idle_days=idle,
        notice=None if notice is None else int(notice),
        open_to_work=None if otw is None or pd.isna(otw) else bool(otw),
        github=gh,
        country=str(row.get("country", "")),
        city=loc.split(",")[0].strip(),
        relocate=None if reloc is None or pd.isna(reloc) else bool(reloc),
        n_employers=int(_num(row.get("n_employers"), 0) or 0),
        avg_tenure_months=_num(row.get("avg_tenure_months")),
        eval_mentioned=bool(_EVAL_WORDS.search(narrative)),
        descriptions=descs,
    )


# ------------------------------------------------------------------ clauses
def _skills_clause(f: Facts, seed: int) -> str | None:
    if not f.skills_named:
        return None
    parts = [f"{s['name']} ({s['proficiency']}, {s['months']} mo)" for s in f.skills_named]
    # neutral lead: the named skills are JD-relevant but not all of them are retrieval skills
    lead = ("JD-relevant skills", "hands-on skills", "listed skills")[seed % 3]
    return f"{lead} including {', '.join(parts[:-1])} and {parts[-1]}" if len(parts) > 1 else f"{lead} including {parts[0]}"


def _positives(f: Facts, seed: int) -> list[str]:
    pos = []
    sk = _skills_clause(f, seed >> 2)
    if sk:
        pos.append(sk)
    rotating = []
    if f.quote:
        rotating.append(f'a role described as "{f.quote}"')
    if f.product_ok:
        rotating.append("product-company experience")
    if 5 <= f.yoe <= 9:
        rotating.append("experience inside the JD's 5-9 year band")
    if rotating:
        pos.append(rotating[seed % len(rotating)])
        if len(pos) < 2 and len(rotating) > 1:
            pos.append(rotating[(seed + 1) % len(rotating)])
    pos = pos[:2]
    if len(pos) == 2 and (seed >> 5) & 1:  # vary which fact comes first
        pos.reverse()
    return pos


def _concerns(f: Facts) -> list[str]:
    # Every concern is a noun phrase so it reads correctly after "Main concern:", "Held back by" or "Caveat:".
    c = []
    if f.resp is not None and f.resp < 0.3:
        c.append(f"a low recruiter response rate ({round(f.resp * 100)}%)")
    if f.idle_days is not None and f.idle_days >= 150:
        c.append(f"about {round(f.idle_days / 30)} months of inactivity (last active {f.last_active})")
    if f.notice is not None and f.notice >= 90:
        c.append(f"a {f.notice}-day notice period against the JD's preference for under 30")
    if f.country and f.country != "India":
        stance = {True: "open to relocating", False: "not open to relocating", None: "relocation not stated"}[f.relocate]
        c.append(f"a base in {f.city or f.country}, {f.country}, while the role is Pune/Noida ({stance})")
    if f.yoe < 5:
        c.append(f"{_yrs(f.yoe)} years of experience, below the JD's 5-9 year band")
    elif f.yoe > 9.5:
        c.append(f"{_yrs(f.yoe)} years of experience, above the JD's 5-9 year band")
    if not f.has_core_skill:
        c.append("no listed retrieval or vector-search skill with meaningful usage time")
    if f.n_employers >= 4 and f.avg_tenure_months is not None and f.avg_tenure_months < 20:
        c.append(f"{f.n_employers} employers at about {round(f.avg_tenure_months)} months each, against the JD's wish for a 3+ year commitment")
    if f.open_to_work is False:
        c.append("no open-to-work flag")
    if f.notice is not None and 60 <= f.notice < 90:
        c.append(f"a {f.notice}-day notice period")
    if not f.product_ok:
        c.append("no product-company experience on the profile")
    if not f.eval_mentioned:
        c.append("no described ranking-evaluation work (NDCG/MRR-style)")
    if f.github is not None and f.github < 0:
        c.append("no linked GitHub activity")
    return c


def _availability_bits(f: Facts) -> str:
    bits = []
    if f.resp is not None:
        bits.append(f"{round(f.resp * 100)}% recruiter response rate")
    if f.notice is not None:
        bits.append("immediate availability" if f.notice == 0 else f"{f.notice}-day notice")
    if f.country:
        bits.append(f"based in {f.city or f.country}")
    return ", ".join(bits)


# ---------------------------------------------------------------- composition
def compose_reasoning(f: Facts, rank: int) -> str:
    seed = _seed(f.candidate_id)
    tier = _tier(rank)
    verdict = _VERDICTS[tier][seed % 3]
    pos = _positives(f, seed >> 3)
    joined = " and ".join(pos) if pos else "no standout retrieval evidence on the profile"
    frame = (seed >> 6) % 5
    yrs = _yrs(f.yoe)
    if frame == 0:
        s1 = f"{verdict}: {f.title} at {f.company} with {yrs} years of experience, showing {joined}."
    elif frame == 1:
        s1 = f"{yrs}-year {f.title} at {f.company} ({verdict.lower()}), with {joined}."
    elif frame == 2:
        s1 = f"{verdict} - {f.company}'s {f.title}, {yrs} years in, with {joined}."
    elif frame == 3:
        s1 = f"{f.title} at {f.company} ({yrs} years): {verdict.lower()}, with {joined}."
    else:
        s1 = f"At {f.company}, a {yrs}-year {f.title} ({verdict.lower()}) with {joined}."

    concerns = _concerns(f)[: 1 if tier == 1 else 2]
    avail = _availability_bits(f)
    frame2 = (seed >> 9) % 4
    if concerns:
        c = "; ".join(concerns)
        s2 = [f"Main concern: {c}.", f"Held back by {c}.", f"Caveat: {c}.", f"The gap to watch: {c}."][frame2]
    elif tier <= 2:
        s2 = [f"No red flags in availability: {avail}.", f"Availability is clean ({avail}).", f"Signals are healthy: {avail}.", f"Availability checks out: {avail}."][frame2]
    else:
        # lower-ranked with nothing to flag: say so honestly -- the rank comes from the relative score, not from a gap
        s2 = [f"No single disqualifying gap, but a lower relative score than higher-ranked profiles; availability: {avail}.",
              f"Nothing disqualifying on record; it sits lower on relative score than the profiles above it. Availability: {avail}."][frame2 % 2]
    return f"{s1} {s2}"


# ----------------------------------------------------------------- slots (for LLM-written connecting words)
def build_slot_parts(f: Facts, rank: int) -> dict:
    """Verified text fragments ("slots") an LLM may place in a sentence WITHOUT being able to change them.
    Returns {"parts": {NAME: exact text}, "required": [names that must appear exactly once]}; optional slots may be omitted."""
    tier = _tier(rank)
    seed = _seed(f.candidate_id)
    parts = {"WHO": f"{f.title} at {f.company}", "YEARS": f"{_yrs(f.yoe)} years of experience", "FIT": _VERDICTS[tier][seed % 3].lower()}
    if f.skills_named:
        names = [f"{s['name']} ({s['proficiency']}, {s['months']} mo)" for s in f.skills_named]
        parts["SKILLS"] = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    if f.quote:
        parts["QUOTE"] = f'"{f.quote}"'
    if f.product_ok:
        parts["PRODUCT"] = "product-company experience"
    concerns = _concerns(f)[: 1 if tier == 1 else 2]
    for i, c in enumerate(concerns, 1):
        parts[f"CONCERN{i}"] = re.sub(r", (below|above) (the JD's [^,;]*)$", r" (\1 \2)", c)  # a comma inside a slot reads like a list break
    if not concerns:
        parts["AVAIL"] = _availability_bits(f)
        if tier >= 3:
            parts["NOTE"] = "nothing disqualifying on record, but a lower relative score than higher-ranked profiles"
    optional = {"PRODUCT"}
    return {"parts": parts, "required": [k for k in parts if k not in optional]}


# --------------------------------------------------------------------- audit
def audit_reasoning(text: str, row: pd.Series, skip_skill_parse: bool = False) -> list[str]:
    """Return a list of problems; an empty list means every checkable claim traces back to the record."""
    problems = []
    f = build_facts(row)
    roles_text = " ".join(f.descriptions)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    owned = {s.get("name"): s for s in _load(row, "skills_json")}
    if not skip_skill_parse:
        # 1) every JD skill named must be in the candidate's own skill list, with the stated proficiency / months.
        #    Titles, company names and quoted text can legitimately contain skill words, so blank them out first,
        #    and match longer skill names first ("Information Retrieval Systems" before "Information Retrieval").
        scan = re.sub(r'"[^"]*"', " ", text)
        for label in (f.title, f.company):
            if label:
                scan = scan.replace(label, " ")
        for name in sorted(_SKILL_GROUP, key=len, reverse=True):
            pat = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])(?: \((\w+), (\d+) mo\))?")
            for m in pat.finditer(scan):
                if name not in owned:
                    problems.append(f"skill not in profile: {name}")
                elif m.group(1):
                    s = owned[name]
                    if s.get("proficiency") != m.group(1) or int(s.get("duration_months") or 0) != int(m.group(2)):
                        problems.append(f"skill detail mismatch: {name}")
            scan = pat.sub(" ", scan)
        # 2) quotes must be verbatim substrings of the role descriptions
        for q in re.findall(r'"([^"]+)"', text):
            if norm(q.rstrip("…")) not in norm(roles_text):
                problems.append(f"quote not verbatim: {q[:40]}")
        # 1b) any "<name> (<proficiency>, <n> mo)" claim must name a skill the candidate really lists,
        #     even if the name is outside the JD skill groups above
        unquoted = re.sub(r'"[^"]*"', " ", text)
        for m in re.finditer(r"\((beginner|intermediate|advanced|expert), (\d+) mo\)", unquoted):
            before = unquoted[: m.start()].rstrip()
            # the skill name sits between the nearest list separator and the parenthetical
            sep = max(((before.rfind(sp), len(sp)) for sp in ("including ", ", ", " and ")), key=lambda x: x[0])
            name = before[sep[0] + sep[1]:].strip() if sep[0] >= 0 else before.strip()
            s = owned.get(name)
            if s is None:
                problems.append(f"skill not in profile: {name}")
            elif s.get("proficiency") != m.group(1) or int(s.get("duration_months") or 0) != int(m.group(2)):
                problems.append(f"skill detail mismatch: {name}")
    # 3) key numbers must match the record
    if f.title not in text or f.company not in text:
        problems.append("title/company missing or wrong")
    no_quotes = re.sub(r'"[^"]*"', " ", text)
    year_mentions = re.findall(r"(\d+(?:\.\d+)?)(?:-year| years)", no_quotes)
    if not year_mentions or any(y != _yrs(f.yoe) for y in year_mentions):
        problems.append("years of experience missing or wrong")
    for pct in re.findall(r"(\d+)% (?:recruiter response|of recruiter)", text):
        if f.resp is None or int(pct) != round(f.resp * 100):
            problems.append(f"response rate mismatch: {pct}%")
    for n in re.findall(r"(\d+)-day notice", text):
        if f.notice is None or int(n) != f.notice:
            problems.append(f"notice mismatch: {n}")
    if "immediate availability" in text and f.notice != 0:
        problems.append("immediate availability claimed but notice != 0")
    if _FIRST_PERSON.search(re.sub(r'"[^"]*"', "", text)):
        problems.append("first-person pronoun outside a quote")
    if re.match(r"\s*Ranked", text):
        problems.append("generic 'Ranked ...' opener")
    return problems


def generate_reasonings_grounded(candidates_df: pd.DataFrame) -> list[str]:
    """One reasoning per row of `candidates_df`, in row order (rank = position, or the `rank` column if present)."""
    out = []
    for idx, (_, row) in enumerate(candidates_df.iterrows()):
        rank = int(row["rank"]) if "rank" in candidates_df.columns and pd.notna(row.get("rank")) else idx + 1
        f = build_facts(row)
        text = compose_reasoning(f, rank)
        if audit_reasoning(text, row):  # should never happen; fall back to the minimal, always-true statement
            text = f"{f.title} at {f.company} with {_yrs(f.yoe)} years of experience; details on the profile were not specific enough to justify more."
        out.append(text)
    return out
