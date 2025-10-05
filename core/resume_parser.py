# core/resume_parser.py
from __future__ import annotations
import os
from pathlib import Path
import docx2txt

import re
from typing import Tuple, List, Dict


try:
    from pypdf import PdfReader
    _HAVE_PYPDF = True
except Exception:
    _HAVE_PYPDF = False

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}\b")
SECTION_HEADINGS = ["experience","work experience","professional experience","education","projects","skills","summary","objective","certifications","achievements","publications"]
NEGATIVE_JD_CUES = ["job description","we are looking for","responsibilities","requirements","about the role","what you will do","what you’ll do","benefits","who you are","nice to have"]
COVER_SALUTATION_RE = re.compile(r"(?im)^\s*dear\b[^\n]*")
COVER_SIGNOFF_RE = re.compile(r"(?im)^\s*(sincerely|best regards|kind regards|yours truly|respectfully)[\s,]*$")
COVER_LETTER_PHRASES = [r"i am writing to", r"i am excited to apply", r"thank you for your consideration", r"the position at", r"i would like to express"]
BULLET_RE = re.compile(r"(?m)^\s*(?:[-•●▪◦–]|[0-9]+\.)\s+")

def classify_document(text: str) -> Tuple[str, Dict[str, List[str] | int]]:
    details: Dict[str, List[str] | int] = {"resume_score": 0, "cover_score": 0, "jd_score": 0,
                                           "resume_signals": [], "cover_signals": [], "jd_signals": []}
    if not text: return "other", details
    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) < 200: return "other", details

    # Resume signals
    rs = 0
    if EMAIL_RE.search(norm): rs += 2; details["resume_signals"].append("email")
    if PHONE_RE.search(norm): rs += 1; details["resume_signals"].append("phone")
    found_sections = [h for h in SECTION_HEADINGS if re.search(rf"(?im)^\s*{re.escape(h)}\b", text)]
    if found_sections: rs += min(4, len(found_sections)); details["resume_signals"] += [f"section:{h}" for h in found_sections]
    if BULLET_RE.findall(text): rs += 1; details["resume_signals"].append("bullets")
    if re.search(r"(?i)\b(B\.?E\.?|B\.?Tech|BSc|MSc|M\.?S\.?|M\.?Tech|MBA|Ph\.?D)\b", norm): rs += 1; details["resume_signals"].append("degree")
    if len(re.findall(r"(?:19|20)\d{2}", norm)) >= 2: rs += 1; details["resume_signals"].append("years timeline")

    # Cover signals
    cs = 0
    if COVER_SALUTATION_RE.search(text): cs += 2; details["cover_signals"].append("salutation")
    if COVER_SIGNOFF_RE.search(text): cs += 1; details["cover_signals"].append("sign-off")
    if any(re.search(p, norm, re.I) for p in COVER_LETTER_PHRASES): cs += 2; details["cover_signals"].append("letter phrasing")
    if re.search(r"(?i)\bcover letter\b", norm): cs += 2; details["cover_signals"].append("mentions 'cover letter'")

    # JD signals
    js = 0
    jd_hits = [kw for kw in NEGATIVE_JD_CUES if re.search(rf"(?i)\b{kw}\b", norm)]
    if jd_hits: js += len(jd_hits); details["jd_signals"] += jd_hits

    details["resume_score"], details["cover_score"], details["jd_score"] = rs, cs, js

    # Strict decision: only pass explicit resume
    if rs >= 5 and rs >= cs + 1 and rs >= js + 1:
        return "resume", details
    if cs >= 3 and cs >= rs and cs >= js:
        return "cover_letter", details
    if js >= 3 and js >= rs and js >= cs:
        return "job_description", details
    return "other",details


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def _read_pdf_text_only(path: str) -> str:
    if not _HAVE_PYPDF:
        raise RuntimeError("Install 'pypdf' for PDF text extraction.")
    parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()

def _try_ocr_pdf(path: str) -> str:
    """
    Lazy OCR: only used if text extraction yields (almost) nothing.
    Requires optional deps: pillow, pdf2image, pytesseract, poppler & tesseract binaries.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        # OCR not available on this host
        return ""

    try:
        images = convert_from_path(path, dpi=200)  # needs poppler installed
        ocr_text = []
        for img in images:
            ocr_text.append(pytesseract.image_to_string(img) or "")
        return "\n".join(ocr_text).strip()
    except Exception:
        return ""

def parse_resume(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    ext = Path(path).suffix.lower()

    if ext == ".txt":
        return _read_txt(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".pdf":
        text = _read_pdf_text_only(path)
        # Heuristic: if almost empty, try OCR once (optional)
        if len(text) < 50:
            ocr = _try_ocr_pdf(path)
            return ocr or text
        return text

    # Best effort fallback
    try:
        return _read_docx(path)
    except Exception:
        return _read_txt(path) if ext in (".md", ".log") else None
