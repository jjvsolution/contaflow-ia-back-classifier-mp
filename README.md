Descarga los modelos dentro del contenedor contaflow_ollama:

```
docker exec -it contaflow_ollama ollama pull nomic-embed-text
docker exec -it contaflow_ollama ollama pull llama3.2
```

Luego valida que ya aparecen:
```
curl -s http://localhost:11434/api/tags
```

### M24-006 — tests core sin Ollama real

`tests/test_m24_006_helpers_classify.py`: `pick_chart_ref`, `build_input_text`, salvage/`_parse_model_json`, y `run_classify` con Ollama mock (≥15 tests). Evidencia: `python -m pytest tests/test_m24_006_helpers_classify.py`.

### M13-009 — sugerencias matches conciliación

`POST /v1/reconcile-matches`: recibe unmatched cartola/asientos y propone pares (`matchType=ai`, confianza). ContaFlow exige aprobación humana. Evidencia: `tests/test_reconcile_matches_m13_009.py`.

### M01-028 — IVA chilena (prompt + post-proceso)

`build_system_prompt` incluye reglas explícitas (19%, `vat_affected` / `vat_exempt`, activo fijo). Tras el modelo, `apply_chilean_vat_postprocess` normaliza `taxTreatment`/`category`/cuenta 1.2.x. Criterio: compra de activo fijo → `vat_affected` (crédito fiscal). Evidencia: `python -m unittest tests.test_chile_vat_m01_028`.

### M01-027 — chat contable conversacional

`POST /v1/chat`: historial + contexto empresa/período/plan + RAG. Responde JSON (`reply` ES-CL, `citedAccounts`, `suggestedEntry` opcional). Siempre `requiresHumanApproval=true` y `registeredJournalEntry=false` (nunca postea asientos). Evidencia: `python -m pytest tests/test_chat_m01_027.py`.

### M01-026 — OCR facturas PDF/imagen

`POST /v1/ocr` (multipart `file`): PDF digital vía **pypdf**; PDF escaneado/imagen vía **Tesseract** (`spa+eng`, Poppler en Docker). Respuesta con `text`, `engine` y `fields` (`rut`, `folio`, `issueDate`, `amountNet`/`amountVat`/`totalAmount`). Evidencia: `python -m unittest tests.test_ocr_m01_026`.

Uso desde la app: el backend expone `POST /uploads/documents/ocr/preview` (JWT) que reenvía a este endpoint y mapea a campos de formulario (M20-004).

```
curl -s -X POST http://localhost:8000/v1/ocr -F "file=@factura.pdf"
```

### M01-019 — validación Pydantic `/v1/classify`

Body tipado con `LlmRequest` / `ClassificationInput` (`app/llm_schemas.py`), alineado a `llm.types.ts`. Payload inválido → **422** con `detail` (loc/msg). Evidencia: `python -m unittest tests.test_classify_validation_m01_019`.

### M01-024 — upsert RAG por sourceId

`POST /v1/learn` hace upsert por `(tenantId, scope, sourceType, sourceId)`: el segundo learn del mismo documento **actualiza** embedding/payload y no duplica filas. Respuesta: `{ ok, id, updated }`. Índice único parcial en migración Prisma. Evidencia: `python -m unittest tests.test_upsert_rag_m01_024`.

### M01-022 — `ragStatus` en classify

La respuesta de classify incluye `ragStatus: ok|degraded|failed` (y `ragExamplesUsed`). Fallo de búsqueda RAG o embedding → `failed`; 0 ejemplos → `degraded`. Evidencia: `python -m unittest tests.test_rag_status_m01_022`.

### M01-023 — ciclo RAG

Backend con `LLM_LEARNING_ENABLED=true` envía ejemplos a `/v1/learn` al confirmar. Classify reutiliza ejemplos vía `search_examples` + ranking por contraparte. Tests: `test_rag_cycle_m01_023.py` (AI) y `llm-learning.service.spec.ts` (back).

### M01-021 — prompt por `purpose`

`build_system_prompt` especializa el system prompt según `classify_purchase|sale|fee|bank_line` (y `suggest_journal_entry`). Evidencia: `python -m unittest tests.test_purpose_prompt_m01_021`.

### M01-020 — `/health/ready` compuesto

`GET /health/ready` (y alias `GET /ready`) valida PostgreSQL + Ollama con modelos chat/embed. Responde **503** si falla alguna dependencia. Evidencia: `python -m unittest tests.test_health_ready_m01_020`.

### M01-018 — logging estructurado

Sin `print` de debug. Logs JSON en stdout con `event`, `requestId`, `latencyMs` (y `llmLatencyMs` cuando aplica). Nivel: `LOG_LEVEL=INFO` (prod) o `DEBUG` (preview de input). Evidencia: `python -m unittest tests.test_logging_m01_018`.

Probar:
``` 
curl -s -X POST http://localhost:8000/v1/classify -H "Content-Type: application/json" -d "{\"requestId\":\"test\",\"purpose\":\"classify_purchase\",\"input\":{\"requestId\":\"test\",\"tenantId\":\"TU-TENANT-UUID\",\"kind\":\"purchase\",\"company\":{\"companyId\":\"TU-COMPANY-UUID\",\"giro\":\"comercio\"},\"period\":{\"companyId\":\"TU-COMPANY-UUID\",\"fiscalYear\":2026,\"month\":1,\"periodId\":\"TU-PERIOD-UUID\",\"isClosed\":false},\"source\":{\"textRaw\":\"factura papel\"},\"options\":{\"mode\":\"classify_only\",\"explain\":true}}}"
```

cd "c:/Users/josep/Documents/UNAB/PROYECTO DE TITULO/workspace/app-contaflow-ia"

# Solo IA (tiene el /ready de Ollama + modelos)
docker compose up -d --build ai-api

curl https://unemployed-caprice-jjvsolutions-7f31f51a.koyeb.app/api/pull -d '{"name":"llama3.2:latest"}'

curl https://unemployed-caprice-jjvsolutions-7f31f51a.koyeb.app/api/pull -d '{"name":"nomic-embed-text:latest"}'