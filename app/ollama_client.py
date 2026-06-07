import json
import time
from typing import Any

import httpx

from app.config import settings


async def ollama_embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        base = settings.ollama_host.rstrip("/")
        errors: list[str] = []

        # Ollama newer endpoint (batch): POST /api/embed { model, input }
        try:
            r = await client.post(
                f"{base}/api/embed",
                json={"model": settings.ollama_embed_model, "input": text},
            )
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings")
                if isinstance(embs, list) and embs and isinstance(embs[0], list):
                    emb = embs[0]
                else:
                    emb = None
                if not emb:
                    raise ValueError("missing embeddings[0] in /api/embed response")
                if len(emb) != settings.embedding_dimensions:
                    raise ValueError(
                        f"embedding size mismatch: got {len(emb)}, "
                        f"expected {settings.embedding_dimensions}"
                    )
                return emb
        except Exception as e:
            errors.append(f"/api/embed: {e!s}")

        # Ollama legacy endpoint: POST /api/embeddings { model, prompt }
        try:
            r = await client.post(
                f"{base}/api/embeddings",
                json={"model": settings.ollama_embed_model, "prompt": text},
            )
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding")
                if not emb:
                    raise ValueError("missing embedding in /api/embeddings response")
                if len(emb) != settings.embedding_dimensions:
                    raise ValueError(
                        f"embedding size mismatch: got {len(emb)}, "
                        f"expected {settings.embedding_dimensions}"
                    )
                return emb
        except Exception as e:
            errors.append(f"/api/embeddings: {e!s}")

        # OpenAI-compatible endpoint (some Ollama setups/proxies expose this):
        # POST /v1/embeddings { model, input }
        try:
            r = await client.post(
                f"{base}/v1/embeddings",
                json={"model": settings.ollama_embed_model, "input": text},
            )
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                data0 = (data.get("data") or [{}])[0] or {}
                emb = data0.get("embedding")
                if not emb:
                    raise ValueError("missing data[0].embedding in /v1/embeddings response")
                if len(emb) != settings.embedding_dimensions:
                    raise ValueError(
                        f"embedding size mismatch: got {len(emb)}, "
                        f"expected {settings.embedding_dimensions}"
                    )
                return emb
        except Exception as e:
            errors.append(f"/v1/embeddings: {e!s}")

        raise RuntimeError(" | ".join(errors))


async def ollama_chat_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
        "format": "json",
    }
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = (data.get("message") or {}).get("content") or "{}"
    latency_ms = int((time.time() - t0) * 1000)

    try:
        parsed = json.loads(msg)
    except json.JSONDecodeError:
        parsed = {}
    return {"json": parsed, "latencyMs": latency_ms, "raw": msg}
