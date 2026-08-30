"""Build stable input text for embeddings / prompts from ClassificationInput-like dicts."""

import re


def extract_bank_counterparty(text: str) -> str | None:
    """Desde glosa de cartola: 'Pago X SpA folio 123' → 'X SpA'."""
    raw = (text or "").strip()
    if not raw:
        return None

    m = re.match(r"^pago\s+(.+?)(?:\s+folio\s+\d+.*)?$", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.match(
        r"^transferencia\s+(?:recibida\s+de|enviada\s+a|a)\s+(.+)$",
        raw,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    m = re.match(r"^(?:abono|ingreso)\s+(.+)$", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def build_input_text(inp: dict) -> str:
    parts: list[str] = []
    kind = inp.get("kind") or ""
    parts.append(f"tipo:{kind}")

    src = inp.get("source") or {}
    raw = (src.get("textRedacted") or src.get("textRaw") or "").strip()
    if raw:
        parts.append(f"texto:{raw[:4000]}")

    st = inp.get("structured") or {}
    counterparty = st.get("counterpartyName")
    if counterparty:
        parts.append(f"contraparte:{counterparty}")
    elif kind == "bank_statement_line":
        bank = st.get("bank") or {}
        memo = (bank.get("memo") or src.get("textRaw") or "").strip()
        cp = extract_bank_counterparty(memo)
        if cp:
            parts.append(f"contraparte:{cp}")
    if st.get("documentNumber"):
        parts.append(f"folio:{st['documentNumber']}")
    if st.get("issueDate"):
        parts.append(f"fecha:{st['issueDate']}")

    totals = st.get("totals") or {}
    tot = totals.get("total")
    if isinstance(tot, dict) and tot.get("amount"):
        parts.append(f"total:{tot['amount']} {tot.get('currency', '')}")

    bank = st.get("bank") or {}
    if bank.get("memo"):
        parts.append(f"memo_banco:{bank['memo']}")
    if bank.get("postedDate"):
        parts.append(f"fecha_mov:{bank['postedDate']}")

    co = inp.get("company") or {}
    if co.get("giro"):
        parts.append(f"giro:{co['giro']}")

    return "\n".join(parts)


def map_kind_to_document_kind(kind: str) -> str:
    return {
        "purchase": "PURCHASE",
        "sale": "SALE",
        "fee": "FEE",
        "bank_statement_line": "BANK_STATEMENT",
    }.get(kind, "PURCHASE")
