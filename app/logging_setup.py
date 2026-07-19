"""Logging estructurado (M01-018): JSON en una línea con requestId / latencyMs."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


def configure_logging() -> None:
    level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    # Evita spam de librerías en INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))
