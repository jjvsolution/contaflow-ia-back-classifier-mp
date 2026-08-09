import logging
import time
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app import ollama_client
from app.classify_engine import normalize_giro, run_classify
from app.config import settings
from app.db import upsert_example
from app.input_text import build_input_text
from app.llm_schemas import LlmRequest
from app.logging_setup import configure_logging, log_event
from app.ocr_service import run_ocr
from app.reconcile_matches import ReconcileMatchesRequest, suggest_reconcile_matches
from app.readiness import composite_ready_check

configure_logging()
logger = logging.getLogger("contaflow.ai.api")

app = FastAPI(title="contaflow-ia-ai", version="0.1.0")


def verify_internal(
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    if settings.internal_token and x_internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


class LearnRequest(BaseModel):
    tenantId: str
    companyId: str | None = None
    giroKey: str
    scope: Literal["GIRO", "COMPANY"]
    documentKind: str
    sourceType: str | None = None
    sourceId: str | None = None
    input: dict[str, Any]
    payload: dict[str, Any]


@app.get("/health")
async def health():
    return {"ok": True}


async def _ready_response(response: Response) -> dict[str, Any]:
    """PostgreSQL + Ollama con modelos; 503 si alguna dependencia falla."""
    payload = await composite_ready_check()
    if not payload.get("ok"):
        response.status_code = 503
    return payload


@app.get("/health/ready")
async def health_ready(response: Response):
    """M01-020: readiness compuesto (Postgres + Ollama/modelos)."""
    return await _ready_response(response)


@app.get("/ready")
async def ready(response: Response):
    """Alias de /health/ready (compatible con Nest GET …/ready)."""
    return await _ready_response(response)


@app.post("/v1/classify")
async def classify_v1(
    body: LlmRequest,
    _: Annotated[None, Depends(verify_internal)],
):
    """Body validado con Pydantic (= LlmRequest / llm.types.ts). Inválido → 422."""
    t0 = time.perf_counter()
    payload = body.model_dump(mode="python")
    request_id = body.input.requestId or body.requestId
    try:
        result = await run_classify(payload)
        return result
    except Exception as e:
        log_event(
            logger,
            "classify_unhandled",
            level=logging.ERROR,
            requestId=request_id,
            latencyMs=int((time.perf_counter() - t0) * 1000),
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"classify failed: {e!s}",
        ) from e

@app.post("/v1/reconcile-matches")
async def reconcile_matches_v1(
    body: ReconcileMatchesRequest,
    _: Annotated[None, Depends(verify_internal)],
):
    """M13-009: propone pares cartola↔asiento entre unmatched (aprobación humana en ContaFlow)."""
    t0 = time.perf_counter()
    matches = suggest_reconcile_matches(body)
    log_event(
        logger,
        "reconcile_matches_done",
        bank=len(body.unmatchedBank),
        journal=len(body.unmatchedJournal),
        suggested=len(matches),
        latencyMs=int((time.perf_counter() - t0) * 1000),
    )
    return {
        "ok": True,
        "matches": matches,
        "requiresHumanApproval": True,
        "provider": {"type": "local", "model": "reconcile-sim-v1"},
    }


@app.post("/v1/ocr")
async def ocr_v1(
    _: Annotated[None, Depends(verify_internal)],
    file: UploadFile = File(..., description="PDF o imagen de factura/boleta"),
):
    """M01-026: extrae texto y campos (RUT, folio, montos, fecha) de PDF/imagen."""
    t0 = time.perf_counter()
    data = await file.read()
    max_bytes = 25 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=400, detail="Archivo supera 25 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    result = run_ocr(
        filename=file.filename or "upload.bin",
        content_type=file.content_type,
        data=data,
    )
    log_event(
        logger,
        "ocr_done",
        filename=file.filename,
        engine=result.get("engine"),
        pageCount=result.get("pageCount"),
        hasRut=bool((result.get("fields") or {}).get("rut")),
        hasFolio=bool((result.get("fields") or {}).get("folio")),
        latencyMs=int((time.perf_counter() - t0) * 1000),
    )
    return result


@app.post("/v1/learn")
async def learn_v1(
    req: LearnRequest,
    _: Annotated[None, Depends(verify_internal)],
):
    t0 = time.perf_counter()
    if req.scope == "COMPANY" and not req.companyId:
        raise HTTPException(
            status_code=400,
            detail="companyId requerido cuando scope=COMPANY",
        )
    if req.scope == "GIRO" and req.companyId:
        raise HTTPException(
            status_code=400,
            detail="companyId debe ser null cuando scope=GIRO",
        )

    inp = req.input
    text = build_input_text(inp)
    try:
        emb = await ollama_client.ollama_embed(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"embedding failed: {e!s}") from e

    result = upsert_example(
        tenant_id=req.tenantId,
        company_id=req.companyId,
        giro_key=normalize_giro(req.giroKey),
        scope=req.scope,
        document_kind=req.documentKind,
        source_type=req.sourceType,
        source_id=req.sourceId,
        input_text=text,
        payload=req.payload,
        embedding=emb,
    )
    log_event(
        logger,
        "learn_done",
        requestId=(inp or {}).get("requestId"),
        tenantId=req.tenantId,
        companyId=req.companyId,
        scope=req.scope,
        exampleId=result["id"],
        updated=result["updated"],
        latencyMs=int((time.perf_counter() - t0) * 1000),
    )
    return {"ok": True, "id": result["id"], "updated": result["updated"]}


def run_sync():
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run_sync()
