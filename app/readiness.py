"""Readiness compuesto: PostgreSQL + Ollama/modelos (M01-020)."""

from __future__ import annotations

from typing import Any

from app import ollama_client
from app.db import postgres_ready_check


async def composite_ready_check() -> dict[str, Any]:
    pg = postgres_ready_check()
    ollama = await ollama_client.ollama_ready_check()

    postgres_ok = pg.get("status") == "up"
    ollama_ok = bool(ollama.get("ok"))
    ready = postgres_ok and ollama_ok

    payload: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "ok": ready,
        "checks": {
            "postgres": pg.get("status"),
            "ollama": ollama.get("ollama"),
            "models": ollama.get("models"),
        },
    }
    errors: list[str] = []
    if not postgres_ok and pg.get("error"):
        errors.append(f"postgres: {pg['error']}")
    if ollama.get("error"):
        errors.append(f"ollama: {ollama['error']}")
    missing = (ollama.get("models") or {}).get("missing") or []
    if missing:
        errors.append(f"models missing: {', '.join(missing)}")
    if errors:
        payload["error"] = "; ".join(errors)
    return payload
