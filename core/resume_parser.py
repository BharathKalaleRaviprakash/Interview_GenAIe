import os

import docx
import fitz
import io
from PIL import Image
import pytesseract

def extract_text_from_docx(file_path: str) -> str | None:
    """Extracts text from docx file."""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        print(f"python-docx extracted {len(text)} characters.")
        return text
    except FileNotFoundError:
        print(f"Error: DOCX file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return None
    
def extract_text_from_pdf_pymupdf(file_path: str) -> str | None:
    """Extracts text from a PDF file using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") or "" #Add null check
        doc.close()
        print(f"pyMuDF extracted {len(text)} characters.")
        return text
    except FileNotFoundError:
        print(f"Error: PDF File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF with pyMuDF {file_path}: {e}")
        return None
    

def ocr_PDF(file_path: str) -> str | None:
    "Extract text from pdf using OCR (Tesseract)"
    print(f"Attempting OCR on {file_path}...")
    text = ""
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to an image (pixmap)
            # Increase DPI for better OCR quality if needed matrix=fitz.Matrix(2,2)
            pix = page.get_pixmap()
            #convert pixmap to PIL Image
            img_data = pix.tobytes("png") #Use PNG for lossless
            img = Image.open(io.BytesIO(img_data))

            #Perform OCR on the image
            try:
                page_text = pytesseract.image_to_string(img, lang='eng') #Specify language
                if page_text:
                    #print(f"OCR extracted text from page {page_num + 1})
                    text += page_text + "\n"
            except pytesseract.TesseractNotFoundError:
                print("\n--- Tesseract OCR Error ---")
                print("Tesseract executable not found. Please ensure:")
                print("1. Tesseract OCR engine is installed on your system.")
                print("2. The 'tesseract' command is in your system's PATH.")
                print("On Windows, you might need to set the path manually in core/resume_parsor.py")
                print("---------------------\n")
                return None
            except Exception as ocr_e:
                print(f"Error during OCR processing on page {page_num + 1}: {ocr_e}")

        doc.close()
        print(f"OCR extraction finished, total characters: {len(text)}")
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found for OCR at {file_path}")
        return None
    except Exception as e:
        print(f"Error opening PDF for OCR {file_path}: {e}")
        return None
    
def parse_resume(file_path: str) -> str | None:

    _, file_extension = os.path.splitext(file_path)
    text = None

    if file_extension.lower() == ".pdf":
        #Try PyMuPDF first
        text = extract_text_from_pdf_pymupdf(file_path)

        #if PyMuPDF fails or gets very little text, try OCR
        if not text or len(text.strip()) < MIN_TEXT_LENGTH_THRESHOLD:
            print(f"Initial PDF text extraction yielded minimal text ({len(text or '')} chars). Falling back to OCR.")
            text_ocr = ocr_PDF(file_path)
            #Prefer OCR text only if its significantly longer/better
            if text_ocr and len(text_ocr.strip()) > len(text or "".strip()):
                print("Using OCR result as it seems more complete.")
                text = text_ocr
            elif text_ocr:
                print("OCR result not used as initial extraction was longer or OCR failed.")
                #Stick with the (potentially short) text from PyMuPDF if OCR didnt add much
            else:
                print("OCR attempt failed or yielded no text.")
                # Stick with original text, even if short
        
        elif file_extension.lower() == ".docx":
            text = extract_text_from_docx(file_path)
        else:
            print(f"Error: Unsupported file type '{file_extension}'. Please use PDF or DOCX.")
            return None
        
        if text and len(text.strip()) > 0:
            print(f"Resume parsed successfully. Total characters {(text)}")
            text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            return text
        else:
            print("Failed to extract meaningful text from resume after all attempts.")
            return None #Return None if even OCR fails or gets nothing substantial