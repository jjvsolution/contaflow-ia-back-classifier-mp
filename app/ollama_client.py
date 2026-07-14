import json
import re
import time
from typing import Any

import httpx

from app.config import settings


def _ollama_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, str) and err.strip():
        return err.strip()
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return None


def _parse_embedding_response(data: dict[str, Any]) -> list[float] | None:
    err = _ollama_error_message(data)
    if err:
        raise ValueError(err)

    embs = data.get("embeddings")
    if isinstance(embs, list) and embs and isinstance(embs[0], list):
        return embs[0]

    emb = data.get("embedding")
    if isinstance(emb, list):
        return emb

    data0 = (data.get("data") or [{}])[0] or {}
    emb = data0.get("embedding")
    if isinstance(emb, list):
        return emb

    return None


async def ollama_embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        base = settings.ollama_host.rstrip("/")
        errors: list[str] = []

        for path, payload in (
            ("/api/embed", {"model": settings.ollama_embed_model, "input": text}),
            ("/api/embeddings", {"model": settings.ollama_embed_model, "prompt": text}),
            ("/v1/embeddings", {"model": settings.ollama_embed_model, "input": text}),
        ):
            try:
                r = await client.post(f"{base}{path}", json=payload)
                if r.status_code == 404 and not r.content:
                    continue
                data = r.json()
                if r.status_code >= 400:
                    msg = _ollama_error_message(data) or r.text.strip() or f"HTTP {r.status_code}"
                    raise ValueError(msg)
                emb = _parse_embedding_response(data)
                if not emb:
                    raise ValueError(f"missing embedding vector in {path} response")
                if len(emb) != settings.embedding_dimensions:
                    raise ValueError(
                        f"embedding size mismatch: got {len(emb)}, "
                        f"expected {settings.embedding_dimensions}"
                    )
                return emb
            except Exception as e:
                if str(e).strip():
                    errors.append(f"{path}: {e!s}")

        raise RuntimeError(" | ".join(errors) if errors else "no embedding endpoint available")


def _salvage_json_object(msg: str) -> dict[str, Any]:
    text = (msg or "").strip()
    if not text:
        return {}

    out: dict[str, Any] = {}
    for key in (
        "category",
        "taxTreatment",
        "primaryAccountName",
        "primaryAccountCode",
        "primaryAccountId",
    ):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
        if match:
            out[key] = match.group(1)

    conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if conf:
        out["confidence"] = float(conf.group(1))

    alts = re.search(r'"alternativeAccountNames"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if alts:
        out["alternativeAccountNames"] = re.findall(r'"([^"]+)"', alts.group(1))

    return out


def _parse_model_json(msg: str) -> dict[str, Any]:
    text = (msg or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return _salvage_json_object(text)


async def ollama_chat_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"num_predict": 512},
    }
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = (data.get("message") or {}).get("content") or "{}"
    latency_ms = int((time.time() - t0) * 1000)
    parsed = _parse_model_json(msg)
    return {"json": parsed, "latencyMs": latency_ms, "raw": msg}


def _model_matches(installed: str, required: str) -> bool:
    inst = installed.lower().strip()
    req = required.lower().strip()
    return inst == req or inst.startswith(f"{req}:")


async def ollama_list_model_names() -> list[str]:
    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    models = data.get("models") or []
    names: list[str] = []
    for m in models:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            names.append(m["name"])
    return names


async def ollama_ready_check() -> dict[str, Any]:
    """Valida Ollama y que existan los modelos de chat y embeddings configurados."""
    chat = settings.ollama_chat_model
    embed = settings.ollama_embed_model
    required = [chat, embed]

    try:
        installed = await ollama_list_model_names()
    except Exception as e:
        return {
            "ok": False,
            "ollama": "down",
            "error": str(e),
            "models": {
                "required": {"chat": chat, "embed": embed},
                "installed": [],
                "missing": required,
                "present": {"chat": False, "embed": False},
            },
        }

    present_chat = any(_model_matches(n, chat) for n in installed)
    present_embed = any(_model_matches(n, embed) for n in installed)
    missing = [m for m, ok in ((chat, present_chat), (embed, present_embed)) if not ok]

    return {
        "ok": len(missing) == 0,
        "ollama": "up",
        "models": {
            "required": {"chat": chat, "embed": embed},
            "installed": installed,
            "missing": missing,
            "present": {"chat": present_chat, "embed": present_embed},
        },
    }
