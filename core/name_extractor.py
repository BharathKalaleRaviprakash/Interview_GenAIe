# core/name_extractor.py
from __future__ import annotations
import re
from typing import Optional, Tuple

_COMMON_TITLE_WORDS = {
    "resume","curriculum","vitae","cv","profile","summary",
    "engineer","developer","scientist","analyst","manager","student",
    "data","software","machine","learning","ai","ml","intern","trainee",
    "senior","junior","lead","principal","consultant","specialist"
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")

def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()

def _is_probable_name(line: str) -> bool:
    """
    Heuristic: 2–4 tokens, alphabetic (allow hyphen/’), mostly Title Case or UPPER, no digits/@,
    and not obviously a title line.
    """
    line = _clean(line)
    if len(line) < 3 or len(line) > 60: return False
    if any(ch.isdigit() for ch in line): return False
    if "@" in line: return False
    tokens = [t for t in re.split(r"[ \t]", line) if t]
    if not (2 <= len(tokens) <= 4): return False
    # reject lines dominated by role words
    low = set(re.sub(r"[^a-z ]","", line.lower()).split())
    if low & _COMMON_TITLE_WORDS:
        # allow if it's clearly a "Name — Title" line; keep left part as name later
        return "—" in line or "-" in line
    # check casing pattern
    def ok_tok(t: str) -> bool:
        t = t.strip(".,-–—'’")
        if not t: return False
        return t.isupper() or t[0].isupper()
    return all(ok_tok(t) for t in tokens)

def _left_of_dash(line: str) -> str:
    return re.split(r"[–—\-|•]", line, 1)[0].strip()

def _name_from_email(email: str) -> Optional[str]:
    """Infer 'First Last' from local-part like 'jahnavi.ashok' or 'ashok_j'."""
    local = email.split("@", 1)[0]
    parts = re.split(r"[._\-+]", local)
    parts = [p for p in parts if p and not p.isdigit()]
    if not parts: return None
    # heuristics: prefer 2 tokens with length >=2
    if len(parts) == 1:
        return parts[0].title()
    # If last token is 1 char (initial), move it to end without dot
    if len(parts[-1]) == 1:
        parts[-1] = parts[-1]
    cand = " ".join(w.title() for w in parts[:3])
    # prune unlikely usernames like 'data.scientist'
    if any(w.lower() in _COMMON_TITLE_WORDS for w in parts):
        return None
    return cand

def extract_candidate_name(resume_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (full_name, first_name) or (None, None) if not found.
    """
    if not resume_text: return None, None
    lines = [l.strip() for l in resume_text.splitlines()]
    # focus on the first ~15 non-empty lines as "header"
    header = [l for l in lines[:40] if l.strip()][:15]

    # 1) Try header lines
    for raw in header:
        line = _clean(raw)
        # remove obvious contact parts before testing
        line_wo = _EMAIL_RE.sub("", line)
        line_wo = _PHONE_RE.sub("", line_wo).strip(" •|-—")
        if not line_wo: continue
        if _is_probable_name(line_wo):
            name = _left_of_dash(line_wo)
            # Ensure it's not just a single token
            toks = [t for t in name.split() if t]
            if len(toks) >= 2:
                first = toks[0].title()
                return name.title(), first

    # 2) Email-based guess (look anywhere in the header)
    header_text = " ".join(header)
    m = _EMAIL_RE.search(header_text)
    if m:
        guess = _name_from_email(m.group(0))
        if guess:
            toks = guess.split()
            first = toks[0].title() if toks else None
            return guess, first

    # 3) Optional: spaCy NER as a last resort (if installed)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(" ".join(lines[:200]))
        # choose earliest PERSON span with 2–4 tokens
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                toks = [t.text for t in ent if t.text.isalpha()]
                if 1 <= len(toks) <= 4:
                    full = " ".join(toks).title()
                    first = toks[0].title()
                    return full, first
    except Exception:
        pass

    return None, None
