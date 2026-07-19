import logging
import time
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from app import ollama_client
from app.classify_engine import normalize_giro, run_classify
from app.config import settings
from app.db import insert_example
from app.input_text import build_input_text
from app.logging_setup import configure_logging, log_event
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
    body: dict[str, Any],
    _: Annotated[None, Depends(verify_internal)],
):
    """Espera el mismo JSON que envía Nest (LlmRequest: requestId, purpose, input, ...)."""
    t0 = time.perf_counter()
    request_id = (body.get("input") or {}).get("requestId") or body.get("requestId")
    try:
        result = await run_classify(body)
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
        raise


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

    insert_example(
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
        latencyMs=int((time.perf_counter() - t0) * 1000),
    )
    return {"ok": True}


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
