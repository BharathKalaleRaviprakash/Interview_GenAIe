# core/resume_parser.py
import os, io
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract

# ---- Config ----
MIN_TEXT_LENGTH_THRESHOLD = int(os.getenv("RESUME_MIN_TEXT_CHARS", "500"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _ensure_tesseract():
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    elif os.name == "nt":
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default).exists():
            pytesseract.pytesseract.tesseract_cmd = default

# ---------------- PDF/DOCX helpers ----------------

def extract_text_from_docx(file_path: str) -> Optional[str]:
    try:
        d = docx.Document(file_path)
        text = "\n".join(p.text for p in d.paragraphs if p.text)
        print(f"python-docx extracted {len(text)} characters.")
        return text or None
    except FileNotFoundError:
        print(f"Error: DOCX file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return None

def extract_text_from_pdf_pymupdf(file_path: str) -> Optional[str]:
    """Text extraction via PyMuPDF (fast)."""
    try:
        out = []
        with fitz.open(file_path) as doc:
            if doc.is_encrypted:
                try:
                    doc.authenticate("")  # try empty password for lightly protected PDFs
                except Exception:
                    pass
            for page in doc:
                out.append(page.get_text("text") or "")
        text = "\n".join(out)
        print(f"PyMuPDF extracted {len(text)} characters.")
        return text or None
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF with PyMuPDF {file_path}: {e}")
        return None

def ocr_pdf(file_path: str, scale: float = 2.0) -> Optional[str]:
    """OCR each page using Tesseract; scale>1 renders higher DPI for better OCR."""
    print(f"Attempting OCR on {file_path}...")
    _ensure_tesseract()
    try:
        out = []
        with fitz.open(file_path) as doc:
            mat = fitz.Matrix(scale, scale)
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    txt = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")
                    if txt:
                        out.append(txt)
                except pytesseract.TesseractNotFoundError:
                    print("Tesseract not found. Set TESSERACT_CMD or install Tesseract OCR.")
                    return None
                except Exception as ocr_e:
                    print(f"OCR error on page {i}: {ocr_e}")
        ocr_text = "\n".join(out).strip()
        print(f"OCR extraction finished, total characters: {len(ocr_text)}")
        return ocr_text or None
    except FileNotFoundError:
        print(f"Error: PDF file not found for OCR at {file_path}")
        return None
    except Exception as e:
        print(f"Error opening PDF for OCR {file_path}: {e}")
        return None

# ---------------- Public API ----------------

def parse_resume(file_path: str) -> Optional[str]:
    """
    Returns normalized resume text or None.
    - PDF: PyMuPDF text, fallback to OCR if too short.
    - DOCX: python-docx
    - TXT: UTF-8 read
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = p.suffix.lower()
    text: Optional[str] = None

    if ext == ".pdf":
        # 1) Try PyMuPDF
        text = extract_text_from_pdf_pymupdf(str(p)) or ""
        # 2) Fallback to OCR if too short
        if len(text.strip()) < MIN_TEXT_LENGTH_THRESHOLD:
            print(f"Initial PDF extraction short ({len(text.strip())} chars). Falling back to OCR...")
            ocr_text = ocr_pdf(str(p), scale=2.0) or ""
            # prefer OCR if it adds more content
            if len(ocr_text.strip()) > len(text.strip()):
                print("Using OCR result (more complete).")
                text = ocr_text
            else:
                print("Keeping PyMuPDF text (longer or OCR not helpful).")

    elif ext == ".docx":
        text = extract_text_from_docx(str(p)) or ""

    elif ext == ".txt":
        text = p.read_text(encoding="utf-8", errors="ignore")

    else:
        print(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")
        return None

    # Final normalization & guard
    text = (text or "").strip()
    if not text:
        print("Failed to extract meaningful text from resume after all attempts.")
        return None

    print(f"Resume parsed successfully. Total characters {len(text)}")
    # collapse excessive blank lines/whitespace
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return normalized or None
