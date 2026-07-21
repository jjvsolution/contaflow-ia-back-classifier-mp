"""M01-019: /v1/classify valida body Pydantic; inválido → 422 con detalle."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.llm_schemas import LlmRequest
from app.main import app, verify_internal


def _valid_body(**overrides):
    body = {
        "requestId": "req-1",
        "purpose": "classify_purchase",
        "input": {
            "requestId": "req-1",
            "tenantId": "t1",
            "kind": "purchase",
            "period": {
                "companyId": "c1",
                "fiscalYear": 2026,
                "month": 1,
                "periodId": "p1",
                "isClosed": False,
            },
            "company": {"companyId": "c1", "giro": "comercio"},
            "source": {"textRaw": "factura"},
            "options": {"mode": "classify_only", "explain": True},
        },
    }
    body.update(overrides)
    return body


class LlmSchemaContractTests(unittest.TestCase):
    def test_payload_valido_parsea(self):
        req = LlmRequest.model_validate(_valid_body())
        self.assertEqual(req.purpose, "classify_purchase")
        self.assertEqual(req.input.kind, "purchase")
        self.assertEqual(req.input.period.month, 1)

    def test_kind_invalido_falla(self):
        bad = _valid_body()
        bad["input"]["kind"] = "invoice"
        with self.assertRaises(ValidationError) as ctx:
            LlmRequest.model_validate(bad)
        self.assertTrue(any("kind" in err["loc"] for err in ctx.exception.errors()))

    def test_month_fuera_de_rango_falla(self):
        bad = _valid_body()
        bad["input"]["period"]["month"] = 13
        with self.assertRaises(ValidationError) as ctx:
            LlmRequest.model_validate(bad)
        errs = ctx.exception.errors()
        self.assertTrue(any("month" in e["loc"] for e in errs))

    def test_purpose_invalido_falla(self):
        with self.assertRaises(ValidationError):
            LlmRequest.model_validate(_valid_body(purpose="classify_xyz"))

    def test_sin_company_falla(self):
        bad = _valid_body()
        del bad["input"]["company"]
        with self.assertRaises(ValidationError) as ctx:
            LlmRequest.model_validate(bad)
        self.assertTrue(any("company" in e["loc"] for e in ctx.exception.errors()))


class ClassifyHttp422Tests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[verify_internal] = lambda: None
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_payload_invalido_retorna_422_con_detalle(self):
        res = self.client.post(
            "/v1/classify",
            json={"requestId": "x", "purpose": "bad", "input": {}},
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json().get("detail")
        self.assertIsInstance(detail, list)
        self.assertGreater(len(detail), 0)
        self.assertIn("loc", detail[0])
        self.assertIn("msg", detail[0])

    def test_payload_valido_llega_al_engine(self):
        mock_result = {
            "requestId": "req-1",
            "json": {"requestId": "req-1", "outcome": "suggested"},
        }
        with patch(
            "app.main.run_classify",
            new=AsyncMock(return_value=mock_result),
        ) as run:
            res = self.client.post("/v1/classify", json=_valid_body())

        self.assertEqual(res.status_code, 200)
        run.assert_awaited_once()
        sent = run.await_args.args[0]
        self.assertEqual(sent["purpose"], "classify_purchase")
        self.assertEqual(sent["input"]["kind"], "purchase")
        self.assertEqual(sent["input"]["company"]["giro"], "comercio")


if __name__ == "__main__":
    unittest.main()
