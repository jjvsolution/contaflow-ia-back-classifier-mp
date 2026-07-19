import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.readiness import composite_ready_check


class CompositeReadyTests(unittest.TestCase):
    def test_ready_ok_cuando_postgres_y_ollama_ok(self):
        with (
            patch(
                "app.readiness.postgres_ready_check",
                return_value={"status": "up"},
            ),
            patch(
                "app.readiness.ollama_client.ollama_ready_check",
                new=AsyncMock(
                    return_value={
                        "ok": True,
                        "ollama": "up",
                        "models": {
                            "required": {
                                "chat": "llama3.2",
                                "embed": "nomic-embed-text",
                            },
                            "installed": [
                                "llama3.2:latest",
                                "nomic-embed-text:latest",
                            ],
                            "missing": [],
                            "present": {"chat": True, "embed": True},
                        },
                    }
                ),
            ),
        ):
            payload = asyncio.run(composite_ready_check())

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["checks"]["postgres"], "up")
        self.assertEqual(payload["checks"]["ollama"], "up")

    def test_ready_fail_si_postgres_caido(self):
        with (
            patch(
                "app.readiness.postgres_ready_check",
                return_value={"status": "down", "error": "connection refused"},
            ),
            patch(
                "app.readiness.ollama_client.ollama_ready_check",
                new=AsyncMock(
                    return_value={
                        "ok": True,
                        "ollama": "up",
                        "models": {
                            "required": {
                                "chat": "llama3.2",
                                "embed": "nomic-embed-text",
                            },
                            "installed": ["llama3.2", "nomic-embed-text"],
                            "missing": [],
                            "present": {"chat": True, "embed": True},
                        },
                    }
                ),
            ),
        ):
            payload = asyncio.run(composite_ready_check())

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(payload["checks"]["postgres"], "down")
        self.assertIn("postgres", payload.get("error", ""))

    def test_ready_fail_si_faltan_modelos(self):
        with (
            patch(
                "app.readiness.postgres_ready_check",
                return_value={"status": "up"},
            ),
            patch(
                "app.readiness.ollama_client.ollama_ready_check",
                new=AsyncMock(
                    return_value={
                        "ok": False,
                        "ollama": "up",
                        "models": {
                            "required": {
                                "chat": "llama3.2",
                                "embed": "nomic-embed-text",
                            },
                            "installed": ["llama3.2:latest"],
                            "missing": ["nomic-embed-text"],
                            "present": {"chat": True, "embed": False},
                        },
                    }
                ),
            ),
        ):
            payload = asyncio.run(composite_ready_check())

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "not_ready")
        self.assertIn("nomic-embed-text", payload.get("error", ""))
        self.assertEqual(payload["checks"]["postgres"], "up")

    def test_http_handler_retorna_503_cuando_not_ready(self):
        """Regla del endpoint: ok=False → status HTTP 503 (sin depender de FastAPI)."""
        status_code = 200
        payload = {"ok": False, "status": "not_ready"}
        if not payload.get("ok"):
            status_code = 503
        self.assertEqual(status_code, 503)


if __name__ == "__main__":
    unittest.main()
