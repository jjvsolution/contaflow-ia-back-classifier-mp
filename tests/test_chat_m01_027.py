"""M01-027: chat contable — ES-CL, contexto empresa/período/plan, sin asientos sin confirmación."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.chat_engine import build_chat_system_prompt, _parse_chat_json, run_chat
from app.main import app, verify_internal


SAMPLE_CHART = [
    {"accountId": "a1", "code": "5.1.01", "name": "Gastos de oficina"},
    {"accountId": "a2", "code": "1.1.01", "name": "Caja"},
    {"accountId": "a3", "code": "2.1.01", "name": "IVA crédito fiscal"},
]


class ChatPromptTests(unittest.TestCase):
    def test_system_prompt_es_cl_y_contexto(self):
        prompt = build_chat_system_prompt(
            company={
                "companyId": "c1",
                "name": "Demo SpA",
                "giro": "Servicios de software",
                "taxId": "76.543.210-3",
            },
            period={
                "periodId": "p1",
                "fiscalYear": 2026,
                "month": 8,
                "status": "OPEN",
                "isClosed": False,
            },
            chart=SAMPLE_CHART,
            examples=[],
        )
        self.assertIn("chile", prompt.lower())
        self.assertIn("espa", prompt.lower())
        self.assertIn("Demo SpA", prompt)
        self.assertIn("agosto 2026", prompt)
        self.assertIn("5.1.01", prompt)
        self.assertIn("NUNCA registres", prompt)
        self.assertIn("confirmaci", prompt.lower())

    def test_parse_cites_plan_accounts(self):
        parsed = _parse_chat_json(
            {
                "reply": "Te sugiero la cuenta 5.1.01 Gastos de oficina.",
                "citedAccountNames": ["5.1.01", "Caja"],
                "suggestedEntry": {
                    "memo": "Compra insumos",
                    "lines": [
                        {"accountName": "5.1.01 - Gastos de oficina", "debit": "10000", "credit": ""},
                        {"accountName": "1.1.01 Caja", "debit": "", "credit": "10000"},
                    ],
                },
            },
            SAMPLE_CHART,
        )
        self.assertIn("Gastos de oficina", parsed["reply"])
        codes = {c.get("code") for c in parsed["citedAccounts"]}
        self.assertIn("5.1.01", codes)
        self.assertIn("1.1.01", codes)
        self.assertEqual(len(parsed["suggestedEntry"]["lines"]), 2)


class ChatEndpointTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[verify_internal] = lambda: None
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_chat_validation_422(self):
        res = self.client.post("/v1/chat", json={"message": "hola"})
        self.assertEqual(res.status_code, 422)

    @patch("app.chat_engine.ollama_client.ollama_embed", new_callable=AsyncMock)
    @patch("app.chat_engine.search_examples", return_value=[])
    @patch("app.chat_engine.ollama_client.ollama_chat_json", new_callable=AsyncMock)
    def test_chat_ok_never_registers_entry(self, mock_chat, _search, mock_embed):
        mock_embed.return_value = [0.1] * 768
        mock_chat.return_value = {
            "json": {
                "reply": "En Chile el IVA general es 19%. Usa la cuenta 2.1.01 si corresponde.",
                "citedAccountNames": ["2.1.01"],
                "suggestedEntry": {
                    "memo": "IVA",
                    "lines": [
                        {
                            "accountName": "2.1.01",
                            "debit": "1900",
                            "credit": "",
                        }
                    ],
                },
            },
            "latencyMs": 12,
            "raw": "{}",
        }

        res = self.client.post(
            "/v1/chat",
            json={
                "tenantId": "t1",
                "message": "¿Cuál es el IVA en Chile?",
                "company": {
                    "companyId": "c1",
                    "name": "Demo SpA",
                    "giro": "Software",
                },
                "period": {
                    "periodId": "p1",
                    "fiscalYear": 2026,
                    "month": 8,
                    "status": "OPEN",
                    "isClosed": False,
                },
                "chartOfAccounts": SAMPLE_CHART,
                "history": [],
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("19%", data["reply"])
        self.assertTrue(data["requiresHumanApproval"])
        self.assertFalse(data["registeredJournalEntry"])
        self.assertIsNotNone(data.get("suggestedEntry"))
        self.assertTrue(any(a.get("code") == "2.1.01" for a in data["citedAccounts"]))


class ChatEngineUnitTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.chat_engine.ollama_client.ollama_embed", new_callable=AsyncMock)
    @patch("app.chat_engine.search_examples", return_value=[])
    @patch("app.chat_engine.ollama_client.ollama_chat_json", new_callable=AsyncMock)
    async def test_run_chat_flags(self, mock_chat, _search, mock_embed):
        mock_embed.return_value = [0.0] * 8
        mock_chat.return_value = {
            "json": {"reply": "Hola, ¿en qué te ayudo con la contabilidad?", "citedAccountNames": []},
            "latencyMs": 1,
        }
        out = await run_chat(
            {
                "tenantId": "t1",
                "message": "hola",
                "company": {"companyId": "c1", "giro": "retail"},
                "chartOfAccounts": SAMPLE_CHART,
            }
        )
        self.assertTrue(out["requiresHumanApproval"])
        self.assertFalse(out["registeredJournalEntry"])
        self.assertIn("contabilidad", out["reply"].lower())


if __name__ == "__main__":
    unittest.main()
