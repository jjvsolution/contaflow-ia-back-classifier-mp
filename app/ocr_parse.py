"""Parser de campos típicos de facturas/boletas chilenas a partir de texto OCR."""

from __future__ import annotations

import re
from typing import Any

from app.ocr_rut import find_valid_ruts, normalize_dash_chars, parse_chile_rut

_MONTHS_ES: dict[str, int] = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _normalize_text(text: str) -> str:
    """Normaliza guiones unicode y espacios raros del OCR/PDF."""
    s = normalize_dash_chars(text or "")
    s = s.replace("\u00a0", " ").replace("\u2009", " ")
    return s


def _clean_amount(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    # 1.234.567,89 o 1234567.89 o $ 1.234.567 o $: 1.477.913
    s = s.replace("$", "").replace(":", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        # miles con punto, decimal con coma
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) <= 2 and len(parts) == 2:
            s = parts[0].replace(".", "") + "." + parts[1]
        else:
            s = s.replace(",", "")
    else:
        # solo puntos: si hay varios, son miles
        if s.count(".") > 1:
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            if len(right) == 3 and left.isdigit():
                s = left + right
    # Normalizar a string entero/decimal sin $; preferimos enteros CLP.
    try:
        val = float(s)
    except ValueError:
        digits = re.sub(r"[^\d.]", "", s)
        if not digits:
            return None
        try:
            val = float(digits)
        except ValueError:
            return None
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


_DATE_PATTERNS = [
    re.compile(
        r"(?i)(?:fecha(?:\s*(?:/|\s)*\s*hora)?(?:\s+de\s+(?:emisi[oó]n|documento))?|"
        r"fecha\s+emisi[oó]n|emisi[oó]n)\s*[:\-]?\s*"
        r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})"
    ),
    re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"),
    re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b"),
]

_DATE_ES = re.compile(
    r"(?i)(?:fecha(?:\s+de\s+emisi[oó]n|\s+emisi[oó]n)?|emisi[oó]n)\s*[:\-]?\s*"
    r"(\d{1,2})\s+de\s+"
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
    r"octubre|noviembre|diciembre)"
    r"\s+(?:de(?:l)?\s+)?(\d{4})"
)


def _normalize_date(d: str, m: str, y: str) -> str | None:
    try:
        day = int(d)
        month = int(m)
        year = int(y)
    except ValueError:
        return None
    if year < 100:
        year += 2000 if year < 70 else 1900
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1990 <= year <= 2100):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def extract_issue_date(text: str) -> str | None:
    t = _normalize_text(text)
    m_es = _DATE_ES.search(t)
    if m_es:
        month = _MONTHS_ES.get(m_es.group(2).lower())
        if month:
            iso = _normalize_date(m_es.group(1), str(month), m_es.group(3))
            if iso:
                return iso
    for pat in _DATE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        iso = _normalize_date(m.group(1), m.group(2), m.group(3))
        if iso:
            return iso
    return None


# Preferir folio de documento; evitar "Res. Ex. N° 83".
_FOLIO_PATTERNS = [
    re.compile(
        r"(?i)(?:factura|boleta)\s+electr[oó]nica\s+"
        r"n\s*[º°o\.]\s*[:#]?\s*(\d{1,12})\b"
    ),
    re.compile(
        r"(?i)\bn\s*[º°o\.]\s*[:#]?\s*(\d{1,12})\b"
        r"(?!\s*de\s+\d)"  # evita Res. Ex. N° 83 de 30/08
    ),
    re.compile(r"(?i)\bfolio\s*[:#]?\s*(\d{1,12})\b"),
]


def extract_folio(text: str) -> str | None:
    t = _normalize_text(text)
    # Descartar resoluciones SII: "Res. Ex. N° 83" / "Res.99"
    cleaned = re.sub(
        r"(?i)res\.?\s*(?:ex\.?)?\s*n\s*[º°o\.]?\s*\d+",
        " ",
        t,
    )
    for pat in _FOLIO_PATTERNS:
        m = pat.search(cleaned)
        if m:
            return m.group(1).lstrip("0") or "0"
    return None


_AMT = (
    r"([\$]?\s*:?\s*\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?"
    r"|\d+(?:[.,]\d{1,2})?)"
)

_AMOUNT_SPECS: list[tuple[str, re.Pattern[str]]] = [
    (
        "amountNet",
        re.compile(rf"(?i)\b(?:neto|monto\s+neto|afecto)\s*[:\-]?\s*{_AMT}"),
    ),
    (
        "amountVat",
        re.compile(
            rf"(?i)\b(?:i\.?\s*v\.?\s*a\.?(?:\s*19\s*%?)?|iva(?:\s*19\s*%?)?)"
            rf"\s*[:\-]?\s*{_AMT}"
        ),
    ),
    (
        "amountExempt",
        re.compile(rf"(?i)\b(?:exento|monto\s+exento)\s*[:\-]?\s*{_AMT}"),
    ),
    (
        "amountGross",
        re.compile(
            rf"(?i)\b(?:total\s+honorarios|honorarios)\s*[:\-]?\s*{_AMT}"
        ),
    ),
    (
        "amountRetention",
        re.compile(
            rf"(?i)\b(?:impto\.?\s*retenido|impuesto\s+retenido|retenci[oó]n)"
            rf"\s*[:\-]?\s*{_AMT}"
        ),
    ),
    (
        "totalAmount",
        re.compile(
            rf"(?i)\b(?:total(?:\s+a\s+pagar)?|monto\s+total|valor\s+total)"
            rf"(?!\s+honorarios)\s*[:\-]?\s*{_AMT}"
        ),
    ),
]


def extract_amounts(text: str) -> dict[str, str | None]:
    t = _normalize_text(text)
    # Evitar que "IMPUESTO ADICIONAL" robe el slot de IVA.
    t_amt = re.sub(r"(?i)impuesto\s+adicional\s*[:\-]?\s*" + _AMT, " ", t)
    out: dict[str, str | None] = {
        "amountNet": None,
        "amountVat": None,
        "amountExempt": None,
        "amountGross": None,
        "amountRetention": None,
        "totalAmount": None,
    }
    for key, pat in _AMOUNT_SPECS:
        m = pat.search(t_amt)
        if m:
            out[key] = _clean_amount(m.group(1))
    # Si hay neto+total y falta IVA, derivar (CLP 19%).
    if out["amountNet"] and out["totalAmount"] and not out["amountVat"]:
        try:
            net = float(out["amountNet"])
            total = float(out["totalAmount"])
            vat = total - net
            if vat > 0:
                out["amountVat"] = (
                    str(int(round(vat)))
                    if abs(vat - round(vat)) < 1e-9
                    else f"{vat:.2f}".rstrip("0").rstrip(".")
                )
        except ValueError:
            pass
    # Factura afecta: exento vacío → "0" si hay IVA.
    if out["amountVat"] and not out["amountExempt"]:
        out["amountExempt"] = "0"
    return out


def _clean_party_name(raw: str | None) -> str | None:
    if not raw:
        return None
    s = re.sub(r"\s+", " ", raw).strip(" -:;,.")
    s = re.sub(r"(?i)\b(rut|r\.?\s*u\.?\s*t\.?)\b.*$", "", s).strip(" -:;,.")
    if len(s) < 3 or len(s) > 180:
        return None
    if re.fullmatch(r"[\d.\-−\s]+", s):
        return None
    return s


def extract_receiver_name(text: str) -> str | None:
    t = _normalize_text(text)
    m = re.search(
        r"(?i)se[nñ]or(?:es|\(es\))?\s*[:\-]?\s*(.+?)(?=\s+R\.?\s*U\.?\s*T\.?|\s+Rut\b|\n)",
        t,
    )
    return _clean_party_name(m.group(1) if m else None)


def extract_issuer_name(text: str) -> str | None:
    t = _normalize_text(text)
    # Honorarios: nombre del prestador es la 1ª línea antes de BOLETA DE HONORARIOS.
    m_bh = re.search(
        r"(?is)^(.{3,120}?)\n+\s*BOLETA\s+DE\s+HONORARIOS",
        t.lstrip(),
    )
    if m_bh:
        return _clean_party_name(m_bh.group(1))
    # Factura: razón social emisor = líneas iniciales hasta Giro / SEÑOR.
    head = t.split("SEÑOR")[0] if re.search(r"(?i)se[nñ]or", t) else t[:400]
    head = re.split(r"(?i)\bgiro\s*:", head)[0]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    if not lines:
        return None
    # Evitar líneas de dirección / email.
    for ln in lines[:3]:
        if re.search(r"(?i)(email|telefono|http|www\.|@|n\s*[º°])", ln):
            continue
        if re.search(r"\d{3,}", ln) and len(ln) > 40:
            continue
        name = _clean_party_name(ln)
        if name:
            return name
    return _clean_party_name(lines[0])


def extract_sii_document_type(text: str) -> str | None:
    t = _normalize_text(text)
    if re.search(
        r"(?i)factura\s+(?:no\s+afecta|exenta)|exenta\s+electr[oó]nica",
        t,
    ):
        return "34"
    if re.search(r"(?i)factura\s+electr[oó]nica", t):
        return "33"
    if re.search(r"(?i)boleta\s+de\s+honorarios", t):
        return "BHE"
    if re.search(r"(?i)boleta\s+electr[oó]nica", t):
        return "39"
    return None


def extract_sii_operation_type(text: str, *, prefer: str = "auto") -> str | None:
    """prefer: 'purchase' | 'sale' | 'auto'."""
    t = _normalize_text(text)
    m_compra = re.search(
        r"(?i)tipo\s+de\s*\n?\s*compra\s*[:\-]?\s*([A-ZÁÉÍÓÚÑa-záéíóúñ ]{3,40})",
        t,
    )
    m_venta = re.search(
        r"(?i)tipo\s+de\s*\n?\s*venta\s*[:\-]?\s*([A-ZÁÉÍÓÚÑa-záéíóúñ ]{3,40})",
        t,
    )

    def _norm(m: re.Match[str] | None) -> str | None:
        if not m:
            return None
        v = re.sub(r"\s+", " ", m.group(1)).strip(" -:;")
        return v[:50] if v else None

    if prefer == "purchase":
        return _norm(m_compra) or _norm(m_venta)
    if prefer == "sale":
        return _norm(m_venta) or _norm(m_compra)
    return _norm(m_compra) or _norm(m_venta)


def extract_retentions(text: str) -> dict[str, str | None]:
    """Honorarios: retención contribuyente vs terceros."""
    t = _normalize_text(text)
    amounts = extract_amounts(t)
    retained = amounts.get("amountRetention")
    out: dict[str, str | None] = {
        "amountRetentionTaxpayer": None,
        "amountRetentionThirdParty": None,
    }
    if not retained:
        return out
    # BHE típica: el receptor retiene → retención contribuyente.
    if re.search(
        r"(?i)receptor\s+de\s+esta\s+boleta\s+debe\s+retener|"
        r"contribuyente\s+receptor|"
        r"impto\.?\s*retenido",
        t,
    ):
        out["amountRetentionTaxpayer"] = retained
        out["amountRetentionThirdParty"] = "0"
    elif re.search(r"(?i)retenci[oó]n\s+(?:de\s+)?terceros|terceros", t):
        out["amountRetentionThirdParty"] = retained
        out["amountRetentionTaxpayer"] = "0"
    else:
        out["amountRetentionTaxpayer"] = retained
        out["amountRetentionThirdParty"] = "0"
    return out


_RUT_LABELED = re.compile(
    r"(?i)(?:r\.?\s*u\.?\s*t\.?|rut)\s*"
    r"(?:emisor|receptor|proveedor|cliente|prestador)?\s*[:\-]?\s*"
    r"(\d{1,2}(?:\.\d{3}){1,2}|\d{7,8})\s*[-−–—]?\s*([\dkK])"
)


def _split_issuer_receiver(text: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Heurística CL: receptor cerca de Señor(es); emisor cerca de FACTURA/BOLETA Nº."""
    t = _normalize_text(text)
    all_ruts = find_valid_ruts(t)
    if not all_ruts:
        return None, None

    receiver: dict[str, str] | None = None
    m_senores = re.search(
        r"(?i)se[nñ]or(?:es|\(es\))?\s*[:\-]?.{0,120}?"
        r"(?:r\.?\s*u\.?\s*t\.?|rut)\s*[:\-]?\s*"
        r"(\d{1,2}(?:\.\d{3}){1,2}|\d{7,8})\s*[-−–—]?\s*([\dkK])",
        t,
        re.DOTALL,
    )
    if m_senores:
        receiver = parse_chile_rut(f"{m_senores.group(1)}-{m_senores.group(2)}")

    issuer: dict[str, str] | None = None
    # Emisor: último RUT etiquetado inmediatamente antes de FACTURA/BOLETA ELECTRONICA.
    m_doc = re.search(r"(?i)(?:factura|boleta)\s+electr[oó]nica", t)
    if m_doc:
        before = t[: m_doc.start()]
        labeled = list(_RUT_LABELED.finditer(before))
        if labeled:
            last = labeled[-1]
            issuer = parse_chile_rut(f"{last.group(1)}-{last.group(2)}")

    # Boleta honorarios: primer RUT suele ser del prestador (emisor).
    if re.search(r"(?i)boleta\s+de\s+honorarios", t) and not issuer:
        m0 = _RUT_LABELED.search(t)
        if m0:
            issuer = parse_chile_rut(f"{m0.group(1)}-{m0.group(2)}")
        if not issuer:
            issuer = all_ruts[0]

    if not issuer and len(all_ruts) >= 2:
        # En facturas SII típicas: 1° receptor, 2° emisor.
        if receiver and all_ruts[0]["rutNormalized"] == receiver["rutNormalized"]:
            issuer = all_ruts[1]
        else:
            issuer = all_ruts[-1]
    if not issuer:
        issuer = all_ruts[0] if not receiver else (
            next(
                (
                    r
                    for r in all_ruts
                    if r["rutNormalized"] != receiver["rutNormalized"]
                ),
                all_ruts[0],
            )
        )

    if not receiver and len(all_ruts) >= 2 and issuer:
        receiver = next(
            (
                r
                for r in all_ruts
                if r["rutNormalized"] != issuer["rutNormalized"]
            ),
            None,
        )

    return issuer, receiver


def extract_primary_rut(text: str) -> dict[str, str] | None:
    issuer, receiver = _split_issuer_receiver(text)
    # Compat: preferir emisor (proveedor/prestador); si no, receptor.
    return issuer or receiver


def parse_invoice_fields(text: str) -> dict[str, Any]:
    """Extrae RUT, folio, montos y fecha desde texto OCR/PDF."""
    issuer, receiver = _split_issuer_receiver(text)
    amounts = extract_amounts(text)
    retentions = extract_retentions(text)
    total = amounts["totalAmount"]
    if not total and amounts.get("amountGross"):
        total = amounts["amountGross"]
    issuer_name = extract_issuer_name(text)
    receiver_name = extract_receiver_name(text)
    sii_type = extract_sii_document_type(text)
    return {
        "rut": (issuer or receiver or {}).get("rutDisplay")
        if (issuer or receiver)
        else None,
        "rutNormalized": (issuer or receiver or {}).get("rutNormalized")
        if (issuer or receiver)
        else None,
        "rutIssuer": issuer["rutDisplay"] if issuer else None,
        "rutIssuerNormalized": issuer["rutNormalized"] if issuer else None,
        "rutReceiver": receiver["rutDisplay"] if receiver else None,
        "rutReceiverNormalized": receiver["rutNormalized"] if receiver else None,
        "issuerName": issuer_name,
        "receiverName": receiver_name,
        "folio": extract_folio(text),
        "issueDate": extract_issue_date(text),
        "siiDocumentType": sii_type,
        "siiOperationType": extract_sii_operation_type(text, prefer="auto"),
        "siiOperationTypePurchase": extract_sii_operation_type(
            text, prefer="purchase"
        ),
        "siiOperationTypeSale": extract_sii_operation_type(text, prefer="sale"),
        "amountNet": amounts["amountNet"],
        "amountVat": amounts["amountVat"],
        "amountExempt": amounts["amountExempt"],
        "amountGross": amounts["amountGross"],
        "amountRetention": amounts["amountRetention"],
        "amountRetentionTaxpayer": retentions["amountRetentionTaxpayer"],
        "amountRetentionThirdParty": retentions["amountRetentionThirdParty"],
        "totalAmount": amounts["totalAmount"] or total,
        "currency": "CLP",
    }
