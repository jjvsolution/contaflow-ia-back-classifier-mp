"""Extracción de texto desde PDF/imagen (pypdf + Tesseract)."""

from __future__ import annotations

import io
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("contaflow.ai.ocr")

EngineName = Literal["pypdf", "tesseract", "none"]

# Umbral: si el PDF trae poco texto, intentamos OCR de página rasterizada.
_MIN_PDF_TEXT_CHARS = 40


@dataclass
class OcrExtractResult:
    text: str
    engine: EngineName
    pageCount: int
    warnings: list[str]


def _tesseract_available() -> bool:
    if shutil.which("tesseract"):
        return True
    # Windows: posibles installs Winget/Chocolatey.
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(candidate):
            os.environ.setdefault("TESSDATA_PREFIX", os.path.dirname(candidate))
            return True
    return False


def _configure_pytesseract() -> None:
    import pytesseract

    cmd = shutil.which("tesseract")
    if not cmd:
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.isfile(candidate):
                cmd = candidate
                break
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _ocr_image_bytes(data: bytes, *, lang: str = "spa+eng") -> str:
    from PIL import Image
    import pytesseract

    _configure_pytesseract()
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    try:
        return pytesseract.image_to_string(img, lang=lang) or ""
    except pytesseract.TesseractError:
        # Fallback si falta spa
        return pytesseract.image_to_string(img, lang="eng") or ""


def _ocr_pil_images(images: list, *, lang: str = "spa+eng") -> str:
    import pytesseract

    _configure_pytesseract()
    parts: list[str] = []
    for img in images:
        try:
            parts.append(pytesseract.image_to_string(img, lang=lang) or "")
        except Exception:
            parts.append(pytesseract.image_to_string(img, lang="eng") or "")
    return "\n".join(p for p in parts if p.strip())


def _extract_pdf_text_layer(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip(), len(reader.pages)


def _pdf_to_images(data: bytes, *, max_pages: int = 3):
    from pdf2image import convert_from_bytes

    return convert_from_bytes(
        data,
        dpi=200,
        first_page=1,
        last_page=max_pages,
        fmt="png",
    )


def extract_text_from_upload(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> OcrExtractResult:
    warnings: list[str] = []
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    is_pdf = name.endswith(".pdf") or "pdf" in ctype
    is_image = (
        name.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp"))
        or ctype.startswith("image/")
    )

    if not data:
        return OcrExtractResult(text="", engine="none", pageCount=0, warnings=["Archivo vacío."])

    if is_pdf:
        text, pages = _extract_pdf_text_layer(data)
        if len(text) >= _MIN_PDF_TEXT_CHARS:
            return OcrExtractResult(
                text=text,
                engine="pypdf",
                pageCount=pages,
                warnings=warnings,
            )

        warnings.append(
            "PDF sin capa de texto útil; intentando OCR con Tesseract."
            if text
            else "PDF sin texto embebido; intentando OCR con Tesseract."
        )
        if not _tesseract_available():
            warnings.append("Tesseract no disponible en el entorno.")
            return OcrExtractResult(
                text=text,
                engine="pypdf" if text else "none",
                pageCount=pages,
                warnings=warnings,
            )
        try:
            images = _pdf_to_images(data)
            ocr_text = _ocr_pil_images(images).strip()
            if ocr_text:
                return OcrExtractResult(
                    text=ocr_text,
                    engine="tesseract",
                    pageCount=len(images) or pages,
                    warnings=warnings,
                )
            warnings.append("OCR no obtuvo texto del PDF rasterizado.")
            return OcrExtractResult(
                text=text,
                engine="tesseract",
                pageCount=pages,
                warnings=warnings,
            )
        except Exception as e:
            logger.warning("pdf_ocr_failed: %s", e)
            warnings.append(f"Fallo OCR PDF: {str(e)[:160]}")
            return OcrExtractResult(
                text=text,
                engine="pypdf" if text else "none",
                pageCount=pages,
                warnings=warnings,
            )

    if is_image:
        if not _tesseract_available():
            return OcrExtractResult(
                text="",
                engine="none",
                pageCount=1,
                warnings=["Tesseract no disponible para OCR de imagen."],
            )
        try:
            ocr_text = _ocr_image_bytes(data).strip()
            return OcrExtractResult(
                text=ocr_text,
                engine="tesseract",
                pageCount=1,
                warnings=warnings
                if ocr_text
                else warnings + ["OCR de imagen no obtuvo texto."],
            )
        except Exception as e:
            logger.warning("image_ocr_failed: %s", e)
            return OcrExtractResult(
                text="",
                engine="none",
                pageCount=1,
                warnings=[f"Fallo OCR imagen: {str(e)[:160]}"],
            )

    return OcrExtractResult(
        text="",
        engine="none",
        pageCount=0,
        warnings=[
            "Tipo de archivo no soportado. Use PDF o imagen (PNG/JPG/WEBP).",
        ],
    )
