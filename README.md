Descarga los modelos dentro del contenedor contaflow_ollama:

```
docker exec -it contaflow_ollama ollama pull nomic-embed-text
docker exec -it contaflow_ollama ollama pull llama3.2
```

Luego valida que ya aparecen:
```
curl -s http://localhost:11434/api/tags
```

Probar:
``` 
curl -s -X POST http://localhost:8000/v1/classify -H "Content-Type: application/json" -d "{\"requestId\":\"test\",\"purpose\":\"classify_purchase\",\"input\":{\"requestId\":\"test\",\"tenantId\":\"TU-TENANT-UUID\",\"kind\":\"purchase\",\"company\":{\"companyId\":\"TU-COMPANY-UUID\",\"giro\":\"comercio\"},\"period\":{\"companyId\":\"TU-COMPANY-UUID\",\"fiscalYear\":2026,\"month\":1,\"periodId\":\"TU-PERIOD-UUID\",\"isClosed\":false},\"source\":{\"textRaw\":\"factura papel\"},\"options\":{\"mode\":\"classify_only\",\"explain\":true}}}"
```