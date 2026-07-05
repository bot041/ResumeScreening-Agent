"""Resume parsing module supporting PDF, DOCX, and TXT formats."""

import os
from pathlib import Path
from typing import Union

import pdfplumber
from docx import Document

from utils import clean_text


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF {file_path}: {exc}") from exc
    return clean_text("\n".join(text_parts))


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n".join(paragraphs))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DOCX {file_path}: {exc}") from exc


def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """Extract text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return clean_text(f.read())
    except Exception as exc:
        raise RuntimeError(f"Failed to read TXT {file_path}: {exc}") from exc


def extract_text(file_path: Union[str, Path]) -> str:
    """Auto-detect file type and extract text."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    elif extension == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def parse_resumes_from_folder(folder_path: Union[str, Path]) -> dict:
    """Parse all supported resumes in a folder and return a mapping of filename to text."""
    folder_path = Path(folder_path)
    parsed = {}
    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            parsed[file_path.name] = extract_text(file_path)
    return parsed
