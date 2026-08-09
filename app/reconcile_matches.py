"""M13-009: sugerencias de matches cartola↔asiento (motor local + contrato Nest)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

_STOP = {
    "de",
    "la",
    "el",
    "los",
    "las",
    "y",
    "a",
    "en",
    "del",
    "por",
    "para",
    "con",
    "un",
    "una",
    "al",
    "pago",
    "transferencia",
    "abono",
    "cargo",
}


class ReconcileBankLine(BaseModel):
    id: str
    description: str = ""
    amount: str | float | int = "0"
    postedDate: str | None = None


class ReconcileJournalLine(BaseModel):
    id: str
    description: str = ""
    netAmount: str | float | int = "0"
    date: str | None = None


class ReconcileMatchesRequest(BaseModel):
    unmatchedBank: list[ReconcileBankLine] = Field(default_factory=list)
    unmatchedJournal: list[ReconcileJournalLine] = Field(default_factory=list)
    dateToleranceDays: int = 5
    amountTolerance: float = 50.0
    minConfidence: float = 0.35
    maxSuggestions: int = 50


def _to_number(n: Any) -> float:
    if n is None:
        return 0.0
    if isinstance(n, (int, float)):
        return float(n)
    s = re.sub(r"[^0-9,.\-]", "", str(n)).replace(",", ".")
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def _tokenize(text: str) -> set[str]:
    raw = unicodedata.normalize("NFD", (text or "").lower())
    raw = "".join(c for c in raw if unicodedata.category(c) != "Mn")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    out: set[str] = set()
    for t in raw.split():
        if len(t) < 3 or t in _STOP:
            continue
        out.add(t)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return 0.0 if union == 0 else inter / union


def _days_between(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    try:
        from datetime import date

        da = date.fromisoformat(a[:10])
        db = date.fromisoformat(b[:10])
        return abs((da - db).days)
    except ValueError:
        return None


def suggest_reconcile_matches(req: ReconcileMatchesRequest) -> list[dict[str, Any]]:
    used_je: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for bl in req.unmatchedBank:
        bl_amt = _to_number(bl.amount)
        bl_tok = _tokenize(bl.description)
        best: dict[str, Any] | None = None

        for je in req.unmatchedJournal:
            if je.id in used_je:
                continue
            je_amt = _to_number(je.netAmount)
            amount_diff = abs(bl_amt - je_amt)
            if amount_diff > req.amountTolerance + 1e-9:
                continue
            d = _days_between(bl.postedDate, je.date)
            if d is not None and d > req.dateToleranceDays:
                continue

            sim = _jaccard(bl_tok, _tokenize(je.description))
            amount_score = 1.0 - min(1.0, amount_diff / max(1.0, req.amountTolerance))
            date_score = 0.7 if d is None else 1.0 - min(1.0, d / max(1, req.dateToleranceDays))
            confidence = round(sim * 0.55 + amount_score * 0.25 + date_score * 0.2, 3)
            if confidence < req.minConfidence:
                continue
            if best is None or confidence > best["confidence"]:
                best = {
                    "matchId": f"{bl.id}:{je.id}",
                    "bankLineId": bl.id,
                    "journalEntryId": je.id,
                    "matchType": "ai",
                    "amountDiff": amount_diff,
                    "dateDiffDays": d,
                    "confidence": confidence,
                    "rationaleShort": (
                        f"Glosas similares (score {sim:.2f})"
                        if sim > 0
                        else "Cercanía de monto/fecha"
                    ),
                }

        if best:
            used_je.add(best["journalEntryId"])
            candidates.append(best)

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[: req.maxSuggestions]
