"""
PDF and image OCR utilities.

This module handles text extraction from PDF files. It attempts native
text extraction first, and falls back to OCR on images when necessary.
"""

import io
import logging
from typing import Optional

import pytesseract
from PIL import Image
from PyPDF2 import PageObject, PdfReader

from src.constants import LOG_FILE_PATH
from src.utils import clean_text, setup_logging

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file, with OCR fallback if plain extraction fails.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Cleaned text extracted from the PDF.
    """
    collected_text = ""

    with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        log.info("Opened PDF for extraction: %s", file_path)

        for idx, page in enumerate(reader.pages):
            try:
                page_content = page.extract_text()
                if page_content:
                    collected_text += page_content
                    log.info("Page %d extracted without OCR.", idx)
                else:
                    log.info("No text found on page %d. Running OCR...", idx)
                    collected_text += extract_text_from_images(page)
            except Exception as exc:
                log.error("Error while processing page %d: %s", idx, exc)

    result = clean_text(collected_text)
    log.info("Completed text extraction for %s", file_path)
    return result


# ---------------------------------------------------------------------
# OCR fallback for page images
# ---------------------------------------------------------------------
def extract_text_from_images(page: PageObject) -> str:
    """
    Run OCR on images contained in a PDF page.

    Args:
        page (PageObject): A PDF page object.

    Returns:
        str: Concatenated OCR results from images.
    """
    text_output = ""

    for image_obj in page.images:
        try:
            image = Image.open(io.BytesIO(image_obj.data))
            ocr_result = pytesseract.image_to_string(image)
            text_output += ocr_result
            log.info("OCR successfully applied to image.")
        except Exception as exc:
            log.error("Failed to run OCR on image: %s", exc)

    return text_output
