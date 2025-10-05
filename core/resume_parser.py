# core/resume_parser.py
from __future__ import annotations
import os
from pathlib import Path
import docx2txt

try:
    from pypdf import PdfReader
    _HAVE_PYPDF = True
except Exception:
    _HAVE_PYPDF = False

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
