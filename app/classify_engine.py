import json
import logging
import re
import time
import uuid
from typing import Any

from app.config import settings
from app.db import search_examples
from app.input_text import build_input_text, map_kind_to_document_kind
from app.logging_setup import log_event
from app import ollama_client

logger = logging.getLogger("contaflow.ai.classify")


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def normalize_giro(giro: str) -> str:
    return " ".join(giro.strip().lower().split())


def pick_chart_ref(name: str, chart: list[dict] | None) -> dict[str, Any]:
    if not chart:
        return {"name": name.strip() or "Sin plan de cuentas"}

    raw = name.strip()
    n = raw.lower()
    n_clean = re.sub(r"\s+", " ", n)
    n_alnum = re.sub(r"[^a-z0-9]", "", n_clean)

    if not n_clean:
        return {"name": "Sin plan de cuentas"}

    # 1) Exact by accountId (sometimes model returns UUID).
    for a in chart:
        account_id = str(a.get("accountId") or "").strip().lower()
        if account_id and account_id == n_clean:
            return {k: v for k, v in a.items() if v is not None}

    # 2) Exact by code.
    for a in chart:
        code = str(a.get("code") or "").strip().lower()
        if code and code == n_clean:
            return {k: v for k, v in a.items() if v is not None}

    # 3) Exact by name.
    for a in chart:
        acc_name = str(a.get("name") or "").strip().lower()
        if acc_name and acc_name == n_clean:
            return {k: v for k, v in a.items() if v is not None}

    # 4) Exact by "code - name" or "code name".
    for a in chart:
        code = str(a.get("code") or "").strip()
        acc_name = str(a.get("name") or "").strip()
        combo = f"{code} - {acc_name}".strip(" -").lower()
        combo2 = f"{code} {acc_name}".strip().lower()
        if n_clean in (combo, combo2):
            return {k: v for k, v in a.items() if v is not None}

    # 5) Fuzzy contains by normalized alnum.
    for a in chart:
        code = str(a.get("code") or "").strip().lower()
        acc_name = str(a.get("name") or "").strip().lower()
        combo = f"{code} {acc_name}".strip()
        combo_alnum = re.sub(r"[^a-z0-9]", "", combo)
        if n_alnum and combo_alnum and (n_alnum in combo_alnum or combo_alnum in n_alnum):
            return {k: v for k, v in a.items() if v is not None}

    return {"name": raw}


def build_system_prompt(
    chart: list[dict] | None,
    examples: list[dict],
    wants_entry: bool,
) -> str:
    chart_txt = ""
    if chart:
        lines = []
        for a in chart[:80]:
            code = a.get("code") or ""
            nm = a.get("name") or ""
            lines.append(f"- {code} {nm}".strip())
        chart_txt = "Plan de cuentas (referencia):\n" + "\n".join(lines)

    ex_txt = ""
    if examples:
        blocks = []
        for ex in examples[:6]:
            pj = ex.get("payloadJson")
            if hasattr(pj, "keys"):
                payload = dict(pj)
            else:
                payload = pj
            blocks.append(
                f"Ejemplo histórico (dist={ex.get('dist', 0):.4f}):\n"
                f"  Contexto: {(ex.get('inputText') or '')[:500]}\n"
                f"  Etiqueta JSON: {json.dumps(payload, ensure_ascii=False)[:1200]}"
            )
        ex_txt = "\n\n".join(blocks)

    entry_instr = ""
    if wants_entry:
        entry_instr = (
            "Incluye journalLines: lista de líneas con accountName, debit (string o vacío), "
            "credit (string o vacío), memo opcional. Usa cuentas exactas del plan, idealmente "
            "en formato 'codigo - nombre'. No incluyas líneas sin monto. El asiento debe cuadrar en CLP."
        )
    else:
        entry_instr = "NO incluyas journalLines; solo clasificación y cuenta sugerida."

    return (
        "Eres un asistente contable para Chile (CLP). Responde SOLO JSON válido, sin markdown. "
        "Campos requeridos: category (snake_case corto), taxTreatment (vat_affected|vat_exempt|unknown), "
        "primaryAccountName (nombre exacto, código o 'codigo - nombre' que exista en el plan si es posible), "
        "alternativeAccountNames (array de strings, opcional), confidence (0..1)."
        f"\n{entry_instr}\n\n{chart_txt}\n\nEjemplos similares del mismo cliente/giro:\n{ex_txt}"
    )


def user_payload(inp: dict, wants_entry: bool) -> str:
    body = {
        "kind": inp.get("kind"),
        "text": build_input_text(inp),
        "structured": inp.get("structured"),
        "wantsJournalEntry": wants_entry,
    }
    return json.dumps(body, ensure_ascii=False)


def parse_amount(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    # Accept values returned by the model like "2500", "2.500", "2500 CLP".
    normalized = re.sub(r"[^0-9,.\-]", "", text)
    if not normalized:
        return 0.0

    if "," in normalized and "." in normalized:
        normalized = normalized.replace(".", "").replace(",", ".")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")

    return float(normalized)


def extract_counterparty(input_text: str) -> str | None:
    for line in input_text.splitlines():
        if line.lower().startswith("contraparte:"):
            value = line.split(":", 1)[1].strip().lower()
            if value:
                return value
    return None


def rank_examples_for_prompt(examples: list[dict], input_text: str) -> list[dict]:
    target_cp = extract_counterparty(input_text)
    if not target_cp:
        return examples

    def score(ex: dict) -> tuple[int, float]:
        txt = str(ex.get("inputText") or "").lower()
        same_counterparty = 1 if f"contraparte:{target_cp}" in txt else 0
        return (same_counterparty, -float(ex.get("dist", 1.0)))

    return sorted(examples, key=score, reverse=True)


def chart_for_prompt(chart: list[dict], kind: str, limit: int = 60) -> list[dict]:
    if not chart:
        return []

    code_prefixes: tuple[str, ...]
    if kind == "purchase":
        code_prefixes = ("5.", "2.1", "1.1", "1.2")
    elif kind == "sale":
        code_prefixes = ("4.", "1.1", "2.1")
    elif kind == "fee":
        code_prefixes = ("5.", "2.1")
    else:
        code_prefixes = ("5.", "4.", "2.1", "1.1")

    filtered = [
        a
        for a in chart
        if str(a.get("code") or "").startswith(code_prefixes)
    ]
    pool = filtered if len(filtered) >= 15 else chart
    return pool[:limit]


def is_valid_classification(raw: dict[str, Any]) -> bool:
    if not raw:
        return False
    category = str(raw.get("category") or "").strip()
    account = str(
        raw.get("primaryAccountName")
        or raw.get("primaryAccountCode")
        or raw.get("primaryAccountId")
        or ""
    ).strip()
    return bool(category) and bool(account)


async def run_classify(body: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = body.get("input") or {}
    request_id = inp.get("requestId") or body.get("requestId") or str(uuid.uuid4())
    purpose = body.get("purpose")

    tenant_id = inp.get("tenantId")
    if not tenant_id:
        log_event(
            logger,
            "classify_error",
            requestId=request_id,
            purpose=purpose,
            code="MISSING_TENANT",
            latencyMs=_elapsed_ms(t0),
        )
        return error_result(request_id, inp, "ClassificationInput.tenantId es requerido para RAG.")

    company = inp.get("company") or {}
    company_id = company.get("companyId")
    giro = company.get("giro") or ""
    giro_key = normalize_giro(giro)
    if not company_id:
        log_event(
            logger,
            "classify_error",
            requestId=request_id,
            purpose=purpose,
            code="MISSING_COMPANY",
            latencyMs=_elapsed_ms(t0),
        )
        return error_result(request_id, inp, "company.companyId es requerido.")

    kind = inp.get("kind") or "purchase"
    doc_kind = map_kind_to_document_kind(kind)
    options = inp.get("options") or {}
    wants_entry = (options.get("mode") or "suggest") == "suggest"
    explain = options.get("explain", True)

    log_event(
        logger,
        "classify_start",
        requestId=request_id,
        purpose=purpose,
        kind=kind,
        companyId=company_id,
    )

    input_text = build_input_text(inp)
    logger.debug(
        json.dumps(
            {
                "event": "classify_input_text",
                "requestId": request_id,
                "chars": len(input_text),
                # Solo en DEBUG: no volcar documento completo en INFO/prod
                "preview": input_text[:240],
            },
            ensure_ascii=False,
        )
    )
    try:
        emb = await ollama_client.ollama_embed(input_text)
    except Exception as e:
        log_event(
            logger,
            "classify_error",
            requestId=request_id,
            purpose=purpose,
            kind=kind,
            code="EMBEDDING_ERROR",
            latencyMs=_elapsed_ms(t0),
            error=str(e),
        )
        return error_result(request_id, inp, f"Embedding error: {e!s}")

    examples: list[dict] = []
    try:
        examples = search_examples(
            tenant_id,
            company_id,
            giro_key,
            doc_kind,
            emb,
            settings.rag_company_limit,
            settings.rag_giro_limit,
        )
        examples = rank_examples_for_prompt(examples, input_text)
    except Exception as e:
        log_event(
            logger,
            "classify_rag_fallback",
            level=logging.WARNING,
            requestId=request_id,
            error=str(e),
        )
        examples = []

    chart = (inp.get("accountingContext") or {}).get("chartOfAccountsTop") or []
    prompt_chart = chart_for_prompt(chart, kind)
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(prompt_chart, examples, wants_entry),
        },
        {"role": "user", "content": user_payload(inp, wants_entry)},
    ]

    raw: dict[str, Any] = {}
    latency = 0
    try:
        for attempt in range(2):
            chat_out = await ollama_client.ollama_chat_json(messages)
            raw = chat_out["json"]
            latency = chat_out["latencyMs"]
            if is_valid_classification(raw):
                break
            if attempt == 0:
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Respuesta incompleta. Devuelve SOLO JSON válido con "
                            "category, taxTreatment y primaryAccountName del plan."
                        ),
                    },
                ]
    except Exception as e:
        log_event(
            logger,
            "classify_error",
            requestId=request_id,
            purpose=purpose,
            kind=kind,
            code="LLM_ERROR",
            latencyMs=_elapsed_ms(t0),
            llmLatencyMs=latency,
            error=str(e),
        )
        return error_result(request_id, inp, f"LLM error: {e!s}")

    if not is_valid_classification(raw):
        log_event(
            logger,
            "classify_error",
            requestId=request_id,
            purpose=purpose,
            kind=kind,
            code="INVALID_MODEL_OUTPUT",
            latencyMs=_elapsed_ms(t0),
            llmLatencyMs=latency,
        )
        return error_result(
            request_id,
            inp,
            "El modelo no devolvió category y primaryAccountName; intente de nuevo.",
        )

    period = inp.get("period") or {}
    period_closed = bool(period.get("isClosed"))

    cat = raw.get("category") or "general"
    tax = raw.get("taxTreatment") or "unknown"
    if tax not in ("vat_affected", "vat_exempt", "unknown"):
        tax = "unknown"

    primary_name = (
        raw.get("primaryAccountId")
        or raw.get("primaryAccountCode")
        or raw.get("primaryAccountName")
        or "Gastos generales"
    )
    primary = pick_chart_ref(str(primary_name), chart)
    alts = []
    for i, name in enumerate(raw.get("alternativeAccountNames") or []):
        if i >= 4:
            break
        ref = pick_chart_ref(str(name), chart)
        alts.append(
            {
                **ref,
                "confidence": {
                    "value": max(0.35, 0.72 - i * 0.08),
                    "label": "medium",
                    "rationaleShort": "Alternativa del modelo",
                },
            }
        )

    conf_val = float(raw.get("confidence") or 0.65)
    conf_val = max(0.0, min(1.0, conf_val))
    conf_label = (
        "high" if conf_val >= 0.8 else "medium" if conf_val >= 0.55 else "low"
    )

    confidence = {
        "value": conf_val,
        "label": conf_label,
        "rationaleShort": "Modelo local + RAG",
    }

    result: dict[str, Any] = {
        "requestId": request_id,
        "kind": kind,
        "outcome": "suggested",
        "provider": {
            "type": "local",
            "model": settings.ollama_chat_model,
            "promptVersion": "rag-v1",
            "latencyMs": latency,
        },
        "classification": {"category": cat, "taxTreatment": tax},
        "suggestedAccount": {"primary": primary, "alternatives": alts or None},
        "confidence": confidence,
        "previewPolicy": {
            "requiresHumanApproval": True,
            "periodIsClosedReadOnly": period_closed,
        },
        "warnings": [],
    }

    if wants_entry:
        lines_raw = raw.get("journalLines")
        if isinstance(lines_raw, list) and lines_raw:
            lines = []
            for row in lines_raw:
                acc_name = str(
                    row.get("accountId")
                    or row.get("accountCode")
                    or row.get("accountName")
                    or ""
                ).strip()
                ref = pick_chart_ref(acc_name, chart)
                line: dict[str, Any] = {"account": ref}
                if row.get("debit"):
                    line["debit"] = {
                        "amount": str(row["debit"]).replace(",", ""),
                        "currency": "CLP",
                    }
                if row.get("credit"):
                    line["credit"] = {
                        "amount": str(row["credit"]).replace(",", ""),
                        "currency": "CLP",
                    }
                if row.get("memo"):
                    line["memo"] = row["memo"]
                lines.append(line)
            deb = sum(parse_amount(l.get("debit", {}).get("amount", 0)) for l in lines)
            cre = sum(parse_amount(l.get("credit", {}).get("amount", 0)) for l in lines)
            has_amounts = any(
                parse_amount(l.get("debit", {}).get("amount", 0)) > 0
                or parse_amount(l.get("credit", {}).get("amount", 0)) > 0
                for l in lines
            )
            balanced = has_amounts and abs(deb - cre) < 0.02
            entry_warnings = []
            if not has_amounts:
                entry_warnings.append("Asiento sin montos; requiere completar debe/haber.")
            if has_amounts and not balanced:
                entry_warnings.append("Asiento no balanceado; revisar montos.")
            result["suggestedEntry"] = {
                "entry": {
                    "date": (
                        (inp.get("structured") or {}).get("issueDate")
                        or (inp.get("structured") or {}).get("bank", {}).get("postedDate")
                        or ""
                    ),
                    "description": f"Sugerencia IA local ({kind})",
                    "lines": lines,
                    "isBalanced": balanced,
                    "warnings": entry_warnings,
                },
                "confidence": confidence,
                "explanation": {
                    "summary": "Sugerencia generada por modelo local con contexto recuperado.",
                    "bullets": [
                        f"Categoría: {cat}",
                        f"Ejemplos RAG usados: {len(examples)}",
                    ],
                    "limitations": [
                        "Verificar políticas internas y normativa vigente.",
                    ],
                }
                if explain
                else None,
            }
        else:
            result["suggestedEntry"] = None
    else:
        result["suggestedEntry"] = None

    total_ms = _elapsed_ms(t0)
    result["provider"]["latencyMs"] = total_ms
    log_event(
        logger,
        "classify_done",
        requestId=request_id,
        purpose=purpose,
        kind=kind,
        outcome=result.get("outcome"),
        latencyMs=total_ms,
        llmLatencyMs=latency,
        ragExamples=len(examples),
        category=cat,
    )
    return {"requestId": body.get("requestId") or request_id, "json": result}


def error_result(request_id: str, inp: dict, msg: str) -> dict[str, Any]:
    kind = inp.get("kind") or "purchase"
    return {
        "requestId": request_id,
        "json": {
            "requestId": request_id,
            "kind": kind,
            "outcome": "error",
            "provider": {
                "type": "local",
                "model": settings.ollama_chat_model,
                "promptVersion": "rag-v1",
            },
            "classification": {"category": "unknown", "taxTreatment": "unknown"},
            "previewPolicy": {
                "requiresHumanApproval": True,
                "periodIsClosedReadOnly": bool((inp.get("period") or {}).get("isClosed")),
            },
            "warnings": [],
            "errors": [{"code": "AI_ERROR", "message": msg}],
        },
    }
