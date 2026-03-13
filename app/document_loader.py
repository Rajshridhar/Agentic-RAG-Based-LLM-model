"""Document loader with full validation pipeline (size, MIME type, PDF integrity,
readability, OCR quality).
"""

import os
import sys

import fitz  # PyMuPDF
import textstat
from langchain_community.document_loaders import PDFPlumberLoader

from app.logger import get_logger

logger = get_logger(__name__)

# python-magic import is platform-aware
if sys.platform == "win32":
    import magic  # python-magic-bin on Windows
else:
    import magic  # python-magic on Linux/Mac


def validate_file_size(file_path: str, max_size_mb: float = 50.0) -> float:
    """Return file size in MB; raise ValueError if it exceeds *max_size_mb*."""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File exceeds size limit: {size_mb:.2f} MB > {max_size_mb} MB")
    return size_mb


def validate_mime_type(file_path: str) -> str:
    """Return MIME type; raise ValueError if not 'application/pdf'."""
    mime = magic.from_file(file_path, mime=True)
    if mime != "application/pdf":
        raise ValueError(f"Invalid MIME type: {mime}")
    return mime


def validate_pdf_integrity(file_path: str) -> int:
    """Return page count; raise ValueError if PDF is corrupt or empty."""
    try:
        doc = fitz.open(file_path)
        page_count = doc.page_count
        if page_count == 0:
            raise ValueError("PDF has zero pages")
        doc.close()
    except Exception as exc:
        raise ValueError(f"Corrupted PDF: {exc}") from exc
    return page_count


def validate_readability(text: str, min_score: float = 18.0) -> float:
    """Return Flesch reading-ease score; raise ValueError if below *min_score*."""
    score = textstat.flesch_reading_ease(text)
    if score < min_score:
        raise ValueError(f"Low readability score: {score:.1f} (minimum: {min_score})")
    return score


def validate_ocr_quality(text: str, min_alpha_ratio: float = 0.7) -> float:
    """Return alpha/space ratio; raise ValueError if below *min_alpha_ratio*.

    A high ratio of alphabetic characters + spaces indicates clean OCR output.
    """
    if not text:
        raise ValueError("Empty text: possible OCR failure")
    alpha_space_count = sum(1 for c in text if c.isalpha() or c.isspace())
    ratio = alpha_space_count / len(text)
    if ratio < min_alpha_ratio:
        raise ValueError(
            f"Poor OCR quality: alpha/space ratio {ratio:.2f} < threshold {min_alpha_ratio}"
        )
    return ratio


def load_and_validate_pdf(pdf_path: str) -> list:
    """Load a PDF file, run all validations, and return LangChain Document objects.

    Validations performed (in order):
    1. File size (<= 50 MB)
    2. MIME type (must be application/pdf)
    3. PDF integrity (no corruption, at least one page)
    4. Readability (Flesch score >= 18 for the first page)
    5. OCR quality (average alpha/space ratio >= 0.7)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. File size
    size_mb = validate_file_size(pdf_path)
    logger.info("File size: %.2f MB", size_mb)

    # 2. MIME type
    mime = validate_mime_type(pdf_path)
    logger.info("MIME type: %s", mime)

    # 3. PDF integrity
    page_count = validate_pdf_integrity(pdf_path)
    logger.info("PDF integrity OK – %d page(s)", page_count)

    # Load documents
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    logger.info("Loaded %d document(s)", len(documents))

    if not documents:
        raise ValueError("No documents loaded from PDF")

    # 4. Readability (check first page)
    readability_score = validate_readability(documents[0].page_content)
    logger.info("Readability score (first page): %.1f", readability_score)

    # 5. OCR quality (average across all pages)
    total_ratio = sum(validate_ocr_quality(doc.page_content) for doc in documents)
    avg_ratio = total_ratio / len(documents)
    logger.info("Average OCR quality ratio: %.2f", avg_ratio)

    return documents
