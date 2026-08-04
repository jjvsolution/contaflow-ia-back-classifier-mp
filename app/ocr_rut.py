"""Validación y normalización de RUT chileno (paridad con chile-rut.ts)."""

from __future__ import annotations

import re


def compute_rut_dv(body_digits: str) -> str:
    total = 0
    mul = 2
    for ch in reversed(body_digits):
        total += int(ch) * mul
        mul = 2 if mul == 7 else mul + 1
    r = 11 - (total % 11)
    if r == 11:
        return "0"
    if r == 10:
        return "K"
    return str(r)


def normalize_dash_chars(value: str) -> str:
    """Unifica guiones unicode típicos de PDF/OCR (U+2212, en-dash, em-dash)."""
    return (
        str(value)
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
    )


def parse_chile_rut(input_value: str | None) -> dict[str, str] | None:
    if not input_value or not str(input_value).strip():
        return None
    s = normalize_dash_chars(str(input_value)).strip().upper()
    s = s.replace(".", "").replace("-", "").replace(" ", "")
    if len(s) < 2:
        return None
    dv = s[-1]
    body = s[:-1]
    if not body.isdigit() or not re.fullmatch(r"[\dK]", dv):
        return None
    if compute_rut_dv(body) != dv:
        return None
    # Formateo con puntos cada 3 dígitos desde la derecha.
    parts: list[str] = []
    rem = body
    while rem:
        parts.append(rem[-3:])
        rem = rem[:-3]
    formatted = ".".join(reversed(parts))
    return {
        "rutNormalized": f"{body}{dv}",
        "rutDisplay": f"{formatted}-{dv}",
    }


_RUT_CANDIDATE = re.compile(
    r"\b(\d{1,2}(?:\.\d{3}){1,2}|\d{7,8})\s*[-−–—]?\s*([\dkK])\b",
)


def find_valid_ruts(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in _RUT_CANDIDATE.finditer(text or ""):
        raw = f"{m.group(1)}-{m.group(2)}"
        parsed = parse_chile_rut(raw)
        if not parsed:
            continue
        key = parsed["rutNormalized"]
        if key in seen:
            continue
        seen.add(key)
        found.append(parsed)
    return found
