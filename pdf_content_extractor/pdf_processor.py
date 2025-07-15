# pdf_processor.py

import re
import logging
import pdfplumber
import fitz  # PyMuPDF
from pix2text import Pix2Text
from io import BytesIO
from tqdm import tqdm

def _clean_and_filter_text(text: str) -> str:
    """
    Applies a series of cleaning steps to the raw text extracted from a PDF.
    This is a helper function internal to the module.

    Args:
        text (str): The raw text content.

    Returns:
        str: The cleaned text.
    """
    # 1. Remove reference sections first, as they contain many patterns that can confuse other filters.
    # This pattern looks for common reference headers and removes everything that follows.
    reference_pattern = r'(?i)\n(References|Bibliography|Reference List)\n.*'
    text = re.sub(reference_pattern, "", text, flags=re.DOTALL)
    
    # 2. Remove page headers/footers (customize this list for your documents)
    # This looks for common patterns like "Chapter 1", "Page 23", etc. at the start/end of lines.
    common_headers_footers = [
        r'^\s*Page\s*\d+\s*$',
        r'^\s*Chapter\s*\d+.*$',
    ]
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in common_headers_footers):
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    
    # 3. Consolidate newlines and spaces
    # Replace three or more newlines with two (to preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # 4. De-hyphenate words broken at the end of a line
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 5. A more careful approach to joining lines: join lines that likely form a single sentence.
    # This joins single newlines that are not preceded by sentence-ending punctuation.
    text = re.sub(r'(?<![.\?!"])\n(?!\n)', ' ', text)
    
    return text.strip()

def process_pdf_with_text(filepath: str, start_page: int, end_page: int) -> str:
    """Extracts text from a PDF using direct text extraction (pdfplumber)."""
    text_content = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            num_pages = len(pdf.pages)
            actual_start = max(0, start_page - 1)
            actual_end = num_pages if end_page == -1 else min(num_pages, end_page)

            for i in tqdm(range(actual_start, actual_end), desc=f"Text-Extracting {os.path.basename(filepath)}"):
                page = pdf.pages[i]
                if page.extract_text():
                    text_content += page.extract_text() + "\n\n"
        return _clean_and_filter_text(text_content)
    except Exception as e:
        logging.error(f"Error processing {filepath} with pdfplumber: {e}")
        return ""

def process_pdf_with_ocr(filepath: str, p2t_instance: Pix2Text, start_page: int, end_page: int, dpi: int) -> str:
    """Extracts text from a PDF using OCR (PyMuPDF + Pix2Text)."""
    if p2t_instance is None:
        raise ValueError("Pix2Text instance is required for OCR method.")
        
    text_content = ""
    try:
        doc = fitz.open(filepath)
        num_pages = len(doc)
        actual_start = max(0, start_page - 1)
        actual_end = num_pages if end_page == -1 else min(num_pages, end_page)

        for page_num in tqdm(range(actual_start, actual_end), desc=f"OCR-Extracting {os.path.basename(filepath)}"):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            img_io = BytesIO(img_data)
            # Each page recognition can be a separate process
            text_content += p2t_instance.recognize(img_io) + "\n\n"
        doc.close()
        return _clean_and_filter_text(text_content)
    except Exception as e:
        logging.error(f"Error processing {filepath} with PyMuPDF/Pix2Text: {e}")
        return ""