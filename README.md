Descarga los modelos dentro del contenedor contaflow_ollama:

```
docker exec -it contaflow_ollama ollama pull nomic-embed-text
docker exec -it contaflow_ollama ollama pull llama3.2
```

Luego valida que ya aparecen:
```
curl -s http://localhost:11434/api/tags
```

### M01-018 — logging estructurado

Sin `print` de debug. Logs JSON en stdout con `event`, `requestId`, `latencyMs` (y `llmLatencyMs` cuando aplica). Nivel: `LOG_LEVEL=INFO` (prod) o `DEBUG` (preview de input). Evidencia: `python -m unittest tests.test_logging_m01_018`.

Probar:
``` 
curl -s -X POST http://localhost:8000/v1/classify -H "Content-Type: application/json" -d "{\"requestId\":\"test\",\"purpose\":\"classify_purchase\",\"input\":{\"requestId\":\"test\",\"tenantId\":\"TU-TENANT-UUID\",\"kind\":\"purchase\",\"company\":{\"companyId\":\"TU-COMPANY-UUID\",\"giro\":\"comercio\"},\"period\":{\"companyId\":\"TU-COMPANY-UUID\",\"fiscalYear\":2026,\"month\":1,\"periodId\":\"TU-PERIOD-UUID\",\"isClosed\":false},\"source\":{\"textRaw\":\"factura papel\"},\"options\":{\"mode\":\"classify_only\",\"explain\":true}}}"
```

cd "c:/Users/josep/Documents/UNAB/PROYECTO DE TITULO/workspace/app-contaflow-ia"

# Solo IA (tiene el /ready de Ollama + modelos)
docker compose up -d --build ai-api
