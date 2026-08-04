"""Orquestación OCR: bytes → texto → campos estructurados."""

from __future__ import annotations

from typing import Any

from app.ocr_engine import extract_text_from_upload
from app.ocr_parse import parse_invoice_fields


def run_ocr(
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> dict[str, Any]:
    extracted = extract_text_from_upload(
        filename=filename,
        content_type=content_type,
        data=data,
    )
    fields = parse_invoice_fields(extracted.text)
    warnings = list(extracted.warnings)
    missing = [
        label
        for label, key in (
            ("RUT", "rut"),
            ("folio", "folio"),
            ("fecha", "issueDate"),
            ("montos", "totalAmount"),
        )
        if not fields.get(key)
        and not (
            key == "totalAmount"
            and (fields.get("amountNet") or fields.get("amountVat"))
        )
    ]
    if missing and extracted.text:
        warnings.append(
            "No se pudieron inferir todos los campos: " + ", ".join(missing) + "."
        )
    elif not extracted.text:
        warnings.append("Sin texto para parsear campos.")

    return {
        "ok": True,
        "engine": extracted.engine,
        "pageCount": extracted.pageCount,
        "text": extracted.text,
        "fields": fields,
        "warnings": warnings,
    }
