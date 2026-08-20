"""Chat contable conversacional (M01-027): ES-CL, contexto empresa/período/plan, sin asientos auto."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from app import ollama_client
from app.classify_engine import compute_rag_status, normalize_giro, pick_chart_ref
from app.config import settings
from app.db import search_examples
from app.logging_setup import log_event

logger = logging.getLogger("contaflow.ai.chat")

_MONTHS_ES = (
    "",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _chart_lines(chart: list[dict[str, Any]], limit: int = 80) -> str:
    lines: list[str] = []
    for a in chart[:limit]:
        code = (a.get("code") or "").strip()
        name = (a.get("name") or "").strip()
        lines.append(f"- {code} {name}".strip())
    return "\n".join(lines)


def build_chat_system_prompt(
    *,
    company: dict[str, Any],
    period: dict[str, Any] | None,
    chart: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    company_name = (company.get("name") or "").strip() or "empresa sin nombre"
    giro = (company.get("giro") or "").strip() or "sin giro"
    tax_id = (company.get("taxId") or "").strip() or "—"

    period_txt = "Período contable: no seleccionado."
    if period:
        month = int(period.get("month") or 0)
        year = period.get("fiscalYear")
        status = period.get("status") or "—"
        closed = period.get("isClosed")
        month_label = _MONTHS_ES[month] if 1 <= month <= 12 else str(month)
        closed_txt = "cerrado" if closed else "abierto"
        period_txt = (
            f"Período contable: {month_label} {year} (estado={status}, {closed_txt})."
        )

    chart_txt = _chart_lines(chart) or "(plan vacío)"
    ex_blocks: list[str] = []
    for ex in examples[:4]:
        pj = ex.get("payloadJson")
        payload = dict(pj) if hasattr(pj, "keys") else pj
        ex_blocks.append(
            f"- Contexto: {(ex.get('inputText') or '')[:400]}\n"
            f"  Etiqueta: {json.dumps(payload, ensure_ascii=False)[:600]}"
        )
    examples_txt = "\n".join(ex_blocks) if ex_blocks else "(sin ejemplos RAG)"

    return (
        "Eres el asistente contable de Contaflow IA para Chile.\n"
        "Responde SIEMPRE en español de Chile (vosotros NO; usa tú/ustedes según tono profesional).\n"
        "Moneda y montos: CLP. IVA típico 19% cuando aplique.\n"
        "Contexto fijo de la conversación:\n"
        f"- Empresa: {company_name} (RUT {tax_id}), giro: {giro}.\n"
        f"- {period_txt}\n"
        "Plan de cuentas (cita códigos/nombres exactos cuando hables de cuentas):\n"
        f"{chart_txt}\n\n"
        "Ejemplos históricos similares (RAG):\n"
        f"{examples_txt}\n\n"
        "REGLAS CRÍTICAS:\n"
        "1) NUNCA registres, postes ni confirmes asientos contables. Solo sugieres.\n"
        "2) Si propones un asiento, deja claro que requiere confirmación humana en Contaflow.\n"
        "3) Usa solo cuentas del plan cuando cites cuentas.\n"
        "4) Sé concreto y breve (máx. ~8 oraciones) salvo que pidan detalle.\n"
        "5) Responde SOLO JSON válido (sin markdown) con esta forma:\n"
        "{"
        '"reply":"texto en español Chile",'
        '"citedAccountNames":["código o nombre del plan",...],'
        '"suggestedEntry":null|'
        '{"memo":"...","lines":[{"accountName":"código - nombre","debit":"","credit":"","memo":""}]}'
        "}\n"
        "Si no hay asiento sugerido, suggestedEntry debe ser null."
    )


def _parse_chat_json(raw: dict[str, Any], chart: list[dict[str, Any]]) -> dict[str, Any]:
    reply = str(raw.get("reply") or "").strip()
    if not reply:
        reply = (
            "No pude armar una respuesta clara. Reformula la pregunta contable, por favor."
        )

    cited_raw = raw.get("citedAccountNames") or raw.get("citedAccounts") or []
    cited: list[dict[str, Any]] = []
    if isinstance(cited_raw, list):
        for item in cited_raw[:8]:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("code") or "").strip()
            else:
                name = str(item).strip()
            if not name:
                continue
            ref = pick_chart_ref(name, chart)
            if ref.get("name"):
                cited.append(
                    {
                        k: v
                        for k, v in {
                            "accountId": ref.get("accountId"),
                            "code": ref.get("code"),
                            "name": ref.get("name"),
                        }.items()
                        if v is not None
                    }
                )

    suggested = None
    se = raw.get("suggestedEntry")
    if isinstance(se, dict) and isinstance(se.get("lines"), list) and se["lines"]:
        lines_out: list[dict[str, Any]] = []
        for line in se["lines"][:12]:
            if not isinstance(line, dict):
                continue
            acc_name = str(
                line.get("accountName")
                or line.get("account")
                or line.get("accountCode")
                or ""
            ).strip()
            ref = pick_chart_ref(acc_name, chart) if acc_name else {}
            lines_out.append(
                {
                    "accountCode": ref.get("code") or line.get("accountCode"),
                    "accountName": ref.get("name") or acc_name or None,
                    "debit": str(line.get("debit") or "") or None,
                    "credit": str(line.get("credit") or "") or None,
                    "memo": str(line.get("memo") or "") or None,
                }
            )
        if lines_out:
            suggested = {
                "memo": str(se.get("memo") or "") or None,
                "lines": lines_out,
            }

    return {
        "reply": reply,
        "citedAccounts": cited,
        "suggestedEntry": suggested,
    }


async def _rag_examples(
    *,
    tenant_id: str,
    company_id: str,
    giro: str,
    message: str,
) -> tuple[list[dict[str, Any]], str]:
    failed = False
    examples: list[dict[str, Any]] = []
    try:
        emb = await ollama_client.ollama_embed(message)
        examples = search_examples(
            tenant_id=tenant_id,
            company_id=company_id,
            giro_key=normalize_giro(giro or "general"),
            document_kind="purchase",
            query_embedding=emb,
            company_limit=3,
            giro_limit=2,
        )
    except Exception as e:
        failed = True
        log_event(
            logger,
            "chat_rag_failed",
            level=logging.WARNING,
            error=str(e),
        )
    return examples, compute_rag_status(failed=failed, examples_count=len(examples))


async def run_chat(body: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    request_id = str(body.get("requestId") or uuid.uuid4())
    tenant_id = str(body.get("tenantId") or "")
    message = str(body.get("message") or "").strip()
    company = body.get("company") or {}
    period = body.get("period")
    chart = list(body.get("chartOfAccounts") or [])
    history = list(body.get("history") or [])[-12:]

    company_id = str(company.get("companyId") or "")
    giro = str(company.get("giro") or "")

    examples, rag_status = await _rag_examples(
        tenant_id=tenant_id,
        company_id=company_id,
        giro=giro,
        message=message,
    )

    system = build_chat_system_prompt(
        company=company if isinstance(company, dict) else {},
        period=period if isinstance(period, dict) else None,
        chart=chart,
        examples=examples,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for h in history:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = str(h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": message})

    try:
        llm = await ollama_client.ollama_chat_json(messages)
        parsed = _parse_chat_json(llm.get("json") or {}, chart)
        latency = int(llm.get("latencyMs") or _elapsed_ms(t0))
        provider = {
            "type": "local",
            "model": settings.ollama_chat_model,
            "latencyMs": latency,
        }
    except Exception as e:
        log_event(
            logger,
            "chat_llm_failed",
            level=logging.ERROR,
            requestId=request_id,
            error=str(e),
            latencyMs=_elapsed_ms(t0),
        )
        raise

    # Garantía de producto: el chat NUNCA registra asientos.
    result = {
        "requestId": request_id,
        "reply": parsed["reply"],
        "citedAccounts": parsed["citedAccounts"],
        "suggestedEntry": parsed["suggestedEntry"],
        "requiresHumanApproval": True,
        "registeredJournalEntry": False,
        "ragStatus": rag_status,
        "provider": provider,
    }

    log_event(
        logger,
        "chat_done",
        requestId=request_id,
        tenantId=tenant_id,
        companyId=company_id,
        ragStatus=rag_status,
        cited=len(result["citedAccounts"]),
        hasSuggestedEntry=bool(result["suggestedEntry"]),
        latencyMs=_elapsed_ms(t0),
    )
    return result
