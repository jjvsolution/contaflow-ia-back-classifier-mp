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


def compute_rag_status(
    *,
    failed: bool,
    examples_count: int,
) -> str:
    """M01-022: ok | degraded | failed."""
    if failed:
        return "failed"
    if examples_count <= 0:
        return "degraded"
    return "ok"


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


def resolve_purpose(body: dict[str, Any], kind: str) -> str:
    """Normaliza purpose del request; si falta, deriva desde kind."""
    raw = str(body.get("purpose") or "").strip().lower()
    allowed = {
        "classify_purchase",
        "classify_sale",
        "classify_fee",
        "classify_bank_line",
        "suggest_journal_entry",
        "suggest_adjustment",
    }
    if raw in allowed:
        return raw
    if kind == "purchase":
        return "classify_purchase"
    if kind == "sale":
        return "classify_sale"
    if kind == "fee":
        return "classify_fee"
    if kind == "bank_statement_line":
        return "classify_bank_line"
    return "classify_purchase"


# M01-028: reglas IVA Chile explícitas en el system prompt.
CHILE_VAT_PROMPT_RULES = (
    "REGLAS IVA CHILE (obligatorias):\n"
    "- Tasa general típica: 19% (vatTypicalRate del contexto si viene informado).\n"
    "- taxTreatment=vat_affected: operación afecta a IVA (crédito fiscal en compras; "
    "débito fiscal en ventas). Usa cuando haya IVA, factura afecta o tasa 19%.\n"
    "- taxTreatment=vat_exempt: operación exenta o no afecta (factura/boleta exenta, "
    "sin IVA, IVA 0%). No inventes IVA.\n"
    "- taxTreatment=unknown: solo si no hay indicios claros de afecto/exento.\n"
    "- COMPRA DE ACTIVO FIJO (maquinaria, equipos, notebook/computador, vehículos, "
    "muebles y útiles, inmuebles, PPE, cuentas 1.2.x): categoría fixed_asset (o similar); "
    "cuenta de activo fijo (1.2.x), NO gasto corriente 5.x; si es factura afecta → "
    "taxTreatment=vat_affected (crédito fiscal IVA del activo). "
    "No trates activo fijo afecto como vat_exempt."
)

PURPOSE_PROMPT_VARIANTS: dict[str, str] = {
    "classify_purchase": (
        "PURPOSE=classify_purchase. Clasifica una COMPRA / factura de proveedor. "
        "Prioriza gasto o costo (cuentas 5.x) salvo activo fijo (1.2.x), "
        "IVA crédito fiscal cuando aplique (vat_affected), "
        "y pasivos/proveedores si el contexto es deuda. "
        "Distingue activo fijo vs gasto corriente; activo fijo afecto → vat_affected."
    ),
    "classify_sale": (
        "PURPOSE=classify_sale. Clasifica una VENTA / factura a cliente. "
        "Prioriza ingresos (cuentas 4.x), IVA débito fiscal cuando aplique (vat_affected), "
        "y clientes/caja según el cobro. No uses cuentas de gasto de compra. "
        "Venta exenta → vat_exempt."
    ),
    "classify_fee": (
        "PURPOSE=classify_fee. Clasifica una BOLETA DE HONORARIOS (prestador de servicio). "
        "Prioriza gasto por honorarios/servicios profesionales (5.x), retención si aparece en el texto, "
        "y tratamiento tributario coherente con honorarios en Chile (a menudo vat_exempt o unknown si no hay IVA SII)."
    ),
    "classify_bank_line": (
        "PURPOSE=classify_bank_line. Clasifica un MOVIMIENTO DE CARTOLA BANCARIA. "
        "Monto negativo ≈ egreso/cargo; positivo ≈ ingreso/abono. "
        "Sugiere cuenta bancaria (1.1) y contrapartida (gasto/ingreso/pasivo) según la glosa. "
        "Evita clasificar como factura de compra/venta salvo evidencia clara. "
        "taxTreatment suele ser unknown salvo glosa con IVA explícito."
    ),
    "suggest_journal_entry": (
        "PURPOSE=suggest_journal_entry. El foco es el ASIENTO contable (debe/haber) en CLP. "
        "Las journalLines son obligatorias, balanceadas, con cuentas del plan, "
        "y memos útiles. La categoría/taxTreatment deben alinear con el asiento propuesto "
        "(incl. IVA 19% / crédito-débito fiscal si aplica)."
    ),
    "suggest_adjustment": (
        "PURPOSE=suggest_adjustment. Sugiere cuentas DEBE y HABER para un AJUSTE contable "
        "(depreciación, amortización o provisión). "
        "Típico: Gasto Depreciación / Depreciación Acumulada; Gasto Amortización / Amortización Acumulada; "
        "Gasto Provisión / Provisión. Devuelve suggestedEntry con 2 líneas balanceadas del plan. "
        "Solo sugerencia; requiere aprobación humana. taxTreatment=unknown salvo indicación explícita."
    ),
}

_FIXED_ASSET_RE = re.compile(
    r"activo\s*fijo|fixed[_\s-]?asset|\bppe\b|"
    r"maquinaria|equipos?(?:\s+computacional)?|notebook|laptop|computador(?:es)?|"
    r"impresora|servidor|hardware|"
    r"veh[ií]culo|camioneta|\bauto\b|\bcami[oó]n\b|"
    r"muebles?\s+y\s+[uú]tiles|mobiliario|herramientas?|"
    r"inmueble|edificio|construcci[oó]n|terreno|"
    r"1\.2\.\d+",
    re.IGNORECASE,
)
_EXEMPT_RE = re.compile(
    r"\bexent[oa]s?\b|\bsin\s+iva\b|\biva\s*0(?:\s*%|\s*por\s*ciento)?\b|"
    r"\bno\s+afect[oa]\b|factura\s+exenta|boleta\s+exenta|neto\s+exento",
    re.IGNORECASE,
)
_VAT_AFFECTED_RE = re.compile(
    r"(?<!sin\s)\biva\b|\b19\s*%|\bafect[oa]s?\b|cr[eé]dito\s+fiscal|d[eé]bito\s+fiscal|"
    r"factura\s+afecta|vat_affected",
    re.IGNORECASE,
)
_FIXED_ASSET_CATEGORY_RE = re.compile(
    r"fixed_?asset|activo_?fijo|ppe|capital_?asset|property_plant",
    re.IGNORECASE,
)


def resolve_vat_typical_rate(inp: dict[str, Any] | None) -> float:
    """Tasa IVA típica Chile; usa accountingContext.rules.vatTypicalRate si viene."""
    rules = ((inp or {}).get("accountingContext") or {}).get("rules") or {}
    raw = rules.get("vatTypicalRate")
    if isinstance(raw, (int, float)) and float(raw) > 0:
        rate = float(raw)
        # Acepta 0.19 o 19
        return rate / 100.0 if rate > 1 else rate
    return 0.19


def looks_like_fixed_asset(
    *,
    text: str,
    category: str,
    primary_name: str,
) -> bool:
    blob = f"{text}\n{category}\n{primary_name}"
    if _FIXED_ASSET_CATEGORY_RE.search(str(category or "")):
        return True
    if _FIXED_ASSET_RE.search(blob):
        return True
    code_m = re.search(r"\b(1\.2(?:\.\d+)*)\b", primary_name or "")
    return bool(code_m)


def prefer_fixed_asset_account(
    primary_name: str,
    chart: list[dict] | None,
    *,
    text: str,
) -> str:
    """Si es activo fijo y el modelo eligió gasto 5.x, prefiere cuenta 1.2 del plan."""
    if not chart:
        return primary_name
    current = pick_chart_ref(primary_name, chart)
    code = str(current.get("code") or "")
    if code.startswith("1.2"):
        return primary_name

    text_l = (text or "").lower()
    scored: list[tuple[int, dict]] = []
    for a in chart:
        acc_code = str(a.get("code") or "")
        if not acc_code.startswith("1.2") or not a.get("name"):
            continue
        name_l = str(a.get("name") or "").lower()
        score = 0
        for token in (
            "notebook",
            "computador",
            "maquinaria",
            "equipo",
            "mueble",
            "vehiculo",
            "vehículo",
            "camioneta",
            "herramienta",
            "edificio",
            "terreno",
            "impresora",
            "servidor",
        ):
            if token in text_l and token in name_l:
                score += 3
            elif token in text_l and token[:5] in name_l:
                score += 1
        if "activo" in name_l or "equipo" in name_l or "mueble" in name_l:
            score += 1
        if score:
            scored.append((score, a))

    if not scored:
        # Cualquier posting 1.2.01.* como fallback suave
        for a in chart:
            acc_code = str(a.get("code") or "")
            if acc_code.startswith("1.2.01") and a.get("name"):
                scored.append((1, a))
                break

    if not scored:
        return primary_name

    scored.sort(key=lambda x: (-x[0], str(x[1].get("code") or "")))
    best = scored[0][1]
    code_b = str(best.get("code") or "").strip()
    name_b = str(best.get("name") or "").strip()
    return f"{code_b} - {name_b}" if code_b else name_b


def apply_chilean_vat_postprocess(
    *,
    category: str,
    tax_treatment: str,
    primary_name: str,
    inp: dict[str, Any],
    purpose: str,
    chart: list[dict] | None,
) -> tuple[str, str, str, list[str]]:
    """
    M01-028: normaliza taxTreatment/categoría/cuenta con reglas IVA Chile.
    Retorna (category, taxTreatment, primaryAccountName, warnings).
    """
    warnings: list[str] = []
    tax = tax_treatment if tax_treatment in ("vat_affected", "vat_exempt", "unknown") else "unknown"
    cat = (category or "general").strip() or "general"
    primary = (primary_name or "").strip() or "Gastos generales"

    text = build_input_text(inp)
    kind = str(inp.get("kind") or "")
    is_purchase = purpose == "classify_purchase" or kind == "purchase"
    is_sale = purpose == "classify_sale" or kind == "sale"
    is_fee = purpose == "classify_fee" or kind == "fee"

    structured = inp.get("structured") or {}
    totals = structured.get("totals") or {}
    tax_amt = parse_amount((totals.get("tax") or {}).get("amount"))
    exempt_amt = parse_amount((totals.get("exempt") or {}).get("amount"))
    net_amt = parse_amount((totals.get("net") or {}).get("amount"))

    rate = resolve_vat_typical_rate(inp)
    rate_pct = int(round(rate * 100)) if abs(rate * 100 - round(rate * 100)) < 1e-9 else round(rate * 100, 2)

    exempt_signal = bool(_EXEMPT_RE.search(text)) or (
        exempt_amt > 0 and tax_amt <= 0 and net_amt > 0
    )
    affected_signal = (tax_amt > 0) or (
        bool(_VAT_AFFECTED_RE.search(text)) and not exempt_signal
    )
    # Neto+total con diferencia ~19% → afecto
    total_amt = parse_amount((totals.get("total") or {}).get("amount"))
    if net_amt > 0 and total_amt > net_amt and tax_amt <= 0:
        implied = (total_amt - net_amt) / net_amt
        if abs(implied - rate) < 0.03:
            affected_signal = True

    is_fa = looks_like_fixed_asset(text=text, category=cat, primary_name=primary)

    if is_fee and tax == "vat_affected" and not affected_signal:
        tax = "vat_exempt" if exempt_signal else "unknown"
        warnings.append("Honorarios: taxTreatment ajustado (sin señal clara de IVA).")

    if exempt_signal and not affected_signal:
        if tax != "vat_exempt":
            tax = "vat_exempt"
            warnings.append("Señal de operación exenta → taxTreatment=vat_exempt.")
    elif affected_signal:
        if tax != "vat_affected":
            tax = "vat_affected"
            warnings.append(f"Señal de IVA (~{rate_pct}%) → taxTreatment=vat_affected.")
    elif is_purchase and is_fa and tax in ("unknown", "vat_exempt"):
        # Criterio M01-028: compra activo fijo afecta sugiere IVA crédito (19%).
        if not exempt_signal:
            tax = "vat_affected"
            warnings.append(
                f"Compra de activo fijo: taxTreatment=vat_affected (IVA Chile ~{rate_pct}%, crédito fiscal)."
            )

    if is_purchase and is_fa:
        if not _FIXED_ASSET_CATEGORY_RE.search(cat):
            cat = "fixed_asset"
            warnings.append("Categoría normalizada a fixed_asset (compra de activo fijo).")
        new_primary = prefer_fixed_asset_account(primary, chart, text=text)
        if new_primary != primary:
            primary = new_primary
            warnings.append("Cuenta preferida de activo fijo (1.2.x) sobre gasto corriente.")

    if is_sale and exempt_signal and not affected_signal:
        tax = "vat_exempt"

    return cat, tax, primary, warnings


def build_system_prompt(
    chart: list[dict] | None,
    examples: list[dict],
    wants_entry: bool,
    purpose: str = "classify_purchase",
    vat_typical_rate: float = 0.19,
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

    # suggest_journal_entry / suggest_adjustment siempre pide asiento; el resto respeta wants_entry.
    force_entry = purpose in ("suggest_journal_entry", "suggest_adjustment")
    if force_entry or wants_entry:
        entry_instr = (
            "Incluye journalLines: lista de líneas con accountName, debit (string o vacío), "
            "credit (string o vacío), memo opcional. Usa cuentas exactas del plan, idealmente "
            "en formato 'codigo - nombre'. No incluyas líneas sin monto. El asiento debe cuadrar en CLP."
        )
    else:
        entry_instr = "NO incluyas journalLines; solo clasificación y cuenta sugerida."

    purpose_block = PURPOSE_PROMPT_VARIANTS.get(
        purpose,
        PURPOSE_PROMPT_VARIANTS["classify_purchase"],
    )

    rate_pct = (
        int(round(vat_typical_rate * 100))
        if abs(vat_typical_rate * 100 - round(vat_typical_rate * 100)) < 1e-9
        else round(vat_typical_rate * 100, 2)
    )
    vat_rate_line = f"vatTypicalRate vigente en este request: {rate_pct}%.\n"

    return (
        "Eres un asistente contable para Chile (CLP). Responde SOLO JSON válido, sin markdown. "
        "Campos requeridos: category (snake_case corto), taxTreatment (vat_affected|vat_exempt|unknown), "
        "primaryAccountName (nombre exacto, código o 'codigo - nombre' que exista en el plan si es posible), "
        "alternativeAccountNames (array de strings, opcional), confidence (0..1)."
        f"\n\n{purpose_block}"
        f"\n{vat_rate_line}{CHILE_VAT_PROMPT_RULES}"
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


def parse_confidence(value: Any, default: float = 0.65) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, (int, float)):
            return float(nested)
        if isinstance(nested, str):
            try:
                return float(nested.strip())
            except ValueError:
                return default
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def parse_amount(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    text = str(value).strip()
    if not text:
        return 0.0

    # Accept values returned by the model like "2500", "2.500", "2500 CLP".
    # Evita basura tipo "4.1-" (código de cuenta truncado / JSON incompleto).
    normalized = re.sub(r"[^0-9,.\-]", "", text)
    if not normalized:
        return 0.0

    if "," in normalized and "." in normalized:
        normalized = normalized.replace(".", "").replace(",", ".")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")

    # Signo solo al inicio; quitar guiones/puntos colgantes al final.
    sign = "-" if normalized.startswith("-") else ""
    body = normalized[1:] if sign else normalized
    body = re.sub(r"-", "", body).rstrip(".")
    if not body or body == ".":
        return 0.0
    normalized = f"{sign}{body}"

    try:
        return float(normalized)
    except ValueError:
        return 0.0


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

    tenant_id = inp.get("tenantId")
    if not tenant_id:
        purpose = resolve_purpose(body, str(inp.get("kind") or "purchase"))
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
        purpose = resolve_purpose(body, str(inp.get("kind") or "purchase"))
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
    purpose = resolve_purpose(body, str(kind))
    doc_kind = map_kind_to_document_kind(kind)
    options = inp.get("options") or {}
    wants_entry = (options.get("mode") or "suggest") == "suggest" or purpose == "suggest_journal_entry"
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
            ragStatus="failed",
        )
        return error_result(
            request_id,
            inp,
            f"Embedding error: {e!s}",
            rag_status="failed",
        )

    examples: list[dict] = []
    rag_failed = False
    rag_error: str | None = None
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
        rag_failed = True
        rag_error = str(e)
        log_event(
            logger,
            "classify_rag_fallback",
            level=logging.WARNING,
            requestId=request_id,
            error=rag_error,
            ragStatus="failed",
        )
        examples = []

    rag_status = compute_rag_status(
        failed=rag_failed,
        examples_count=len(examples),
    )

    chart = (inp.get("accountingContext") or {}).get("chartOfAccountsTop") or []
    prompt_chart = chart_for_prompt(chart, kind)
    vat_rate = resolve_vat_typical_rate(inp)
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                prompt_chart,
                examples,
                wants_entry,
                purpose=purpose,
                vat_typical_rate=vat_rate,
            ),
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
    cat, tax, primary_name, vat_warnings = apply_chilean_vat_postprocess(
        category=str(cat),
        tax_treatment=str(tax),
        primary_name=str(primary_name),
        inp=inp,
        purpose=purpose,
        chart=chart,
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

    conf_val = parse_confidence(raw.get("confidence"), 0.65)
    conf_val = max(0.0, min(1.0, conf_val))
    conf_label = (
        "high" if conf_val >= 0.8 else "medium" if conf_val >= 0.55 else "low"
    )

    confidence = {
        "value": conf_val,
        "label": conf_label,
        "rationaleShort": "Modelo local + RAG + reglas IVA Chile",
    }

    warnings: list[str] = []
    if rag_status == "failed":
        warnings.append(
            f"RAG falló; clasificación sin ejemplos históricos ({rag_error or 'error'})."
        )
    elif rag_status == "degraded":
        warnings.append(
            "RAG degradado: no hay ejemplos previos para este tenant/giro/tipo."
        )
    warnings.extend(vat_warnings)

    result: dict[str, Any] = {
        "requestId": request_id,
        "kind": kind,
        "outcome": "suggested",
        "ragStatus": rag_status,
        "ragExamplesUsed": len(examples),
        "provider": {
            "type": "local",
            "model": settings.ollama_chat_model,
            "promptVersion": "rag-v1-cl-vat",
            "latencyMs": latency,
        },
        "classification": {"category": cat, "taxTreatment": tax},
        "suggestedAccount": {"primary": primary, "alternatives": alts or None},
        "confidence": confidence,
        "previewPolicy": {
            "requiresHumanApproval": True,
            "periodIsClosedReadOnly": period_closed,
        },
        "warnings": warnings,
    }

    if wants_entry:
        lines_raw = raw.get("journalLines")
        if isinstance(lines_raw, list) and lines_raw:
            lines = []
            for row in lines_raw:
                if not isinstance(row, dict):
                    continue
                acc_name = str(
                    row.get("accountId")
                    or row.get("accountCode")
                    or row.get("accountName")
                    or ""
                ).strip()
                ref = pick_chart_ref(acc_name, chart)
                line: dict[str, Any] = {"account": ref}
                debit_raw = row.get("debit")
                credit_raw = row.get("credit")
                if isinstance(debit_raw, dict):
                    debit_amt = debit_raw.get("amount")
                else:
                    debit_amt = debit_raw
                if isinstance(credit_raw, dict):
                    credit_amt = credit_raw.get("amount")
                else:
                    credit_amt = credit_raw
                if debit_amt not in (None, "", 0, "0"):
                    line["debit"] = {
                        "amount": str(debit_amt).replace(",", ""),
                        "currency": "CLP",
                    }
                if credit_amt not in (None, "", 0, "0"):
                    line["credit"] = {
                        "amount": str(credit_amt).replace(",", ""),
                        "currency": "CLP",
                    }
                if row.get("memo"):
                    line["memo"] = row["memo"]
                lines.append(line)
            deb = sum(
                parse_amount((l.get("debit") or {}).get("amount", 0)) for l in lines
            )
            cre = sum(
                parse_amount((l.get("credit") or {}).get("amount", 0)) for l in lines
            )
            has_amounts = any(
                parse_amount((l.get("debit") or {}).get("amount", 0)) > 0
                or parse_amount((l.get("credit") or {}).get("amount", 0)) > 0
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
                        or ((inp.get("structured") or {}).get("bank") or {}).get(
                            "postedDate"
                        )
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
        ragStatus=rag_status,
        ragExamples=len(examples),
        category=cat,
    )
    return {"requestId": body.get("requestId") or request_id, "json": result}


def error_result(
    request_id: str,
    inp: dict,
    msg: str,
    *,
    rag_status: str | None = None,
) -> dict[str, Any]:
    kind = inp.get("kind") or "purchase"
    payload: dict[str, Any] = {
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
    }
    if rag_status:
        payload["ragStatus"] = rag_status
    return {"requestId": request_id, "json": payload}
