Descarga los modelos dentro del contenedor contaflow_ollama:

```
docker exec -it contaflow_ollama ollama pull nomic-embed-text
docker exec -it contaflow_ollama ollama pull llama3.2
```

Luego valida que ya aparecen:
```
curl -s http://localhost:11434/api/tags
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
