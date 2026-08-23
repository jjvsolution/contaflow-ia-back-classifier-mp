"""M24-006: pick_chart_ref, build_input_text, salvage JSON, classify con Ollama mock (≥15 tests)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.classify_engine import pick_chart_ref, run_classify
from app.input_text import build_input_text, map_kind_to_document_kind
from app.ollama_client import _parse_model_json, _salvage_json_object

CHART = [
    {
        "accountId": "acc-arriendo",
        "code": "5.1.01.001",
        "name": "Arriendos",
    },
    {
        "accountId": "acc-caja",
        "code": "1.1.01.001",
        "name": "Caja",
    },
    {
        "accountId": "acc-iva",
        "code": "1.2.01.001",
        "name": "IVA crédito fiscal",
    },
]


def _classify_body(**overrides) -> dict:
    body = {
        "requestId": "req-m24-006",
        "purpose": "classify_purchase",
        "input": {
            "requestId": "req-m24-006",
            "tenantId": "tenant-1",
            "kind": "purchase",
            "period": {
                "companyId": "co-1",
                "fiscalYear": 2026,
                "month": 4,
                "periodId": "pe-1",
                "isClosed": False,
            },
            "company": {"companyId": "co-1", "giro": "Servicios de ingeniería"},
            "source": {
                "textRaw": "Factura de arriendo oficina con IVA 19%",
            },
            "structured": {
                "counterpartyName": "Inmobiliaria Demo",
                "documentNumber": "1234",
                "issueDate": "2026-04-10",
                "totals": {"total": {"amount": "119000", "currency": "CLP"}},
            },
            "accountingContext": {"chartOfAccountsTop": CHART},
            "options": {"mode": "classify_only", "explain": True},
        },
    }
    body.update(overrides)
    return body


class PickChartRefTests(unittest.TestCase):
    def test_sin_chart_devuelve_nombre(self):
        self.assertEqual(pick_chart_ref("Arriendos", None), {"name": "Arriendos"})
        self.assertEqual(
            pick_chart_ref("  ", []),
            {"name": "Sin plan de cuentas"},
        )

    def test_match_por_account_id(self):
        ref = pick_chart_ref("acc-arriendo", CHART)
        self.assertEqual(ref["code"], "5.1.01.001")
        self.assertEqual(ref["name"], "Arriendos")

    def test_match_por_codigo(self):
        ref = pick_chart_ref("1.1.01.001", CHART)
        self.assertEqual(ref["accountId"], "acc-caja")

    def test_match_por_nombre(self):
        ref = pick_chart_ref("Arriendos", CHART)
        self.assertEqual(ref["accountId"], "acc-arriendo")

    def test_match_por_code_dash_name(self):
        ref = pick_chart_ref("5.1.01.001 - Arriendos", CHART)
        self.assertEqual(ref["accountId"], "acc-arriendo")

    def test_match_fuzzy_alnum(self):
        # Subcadena alfanumérica (no exact name/code).
        ref = pick_chart_ref("riendos", CHART)
        self.assertEqual(ref["code"], "5.1.01.001")

    def test_sin_match_conserva_nombre_raw(self):
        ref = pick_chart_ref("Cuenta inventada XYZ", CHART)
        self.assertEqual(ref, {"name": "Cuenta inventada XYZ"})


class BuildInputTextTests(unittest.TestCase):
    def test_incluye_tipo_y_texto(self):
        text = build_input_text(
            {
                "kind": "purchase",
                "source": {"textRaw": "factura papel"},
            }
        )
        self.assertIn("tipo:purchase", text)
        self.assertIn("texto:factura papel", text)

    def test_prefiere_text_redacted(self):
        text = build_input_text(
            {
                "kind": "sale",
                "source": {"textRaw": "secreto", "textRedacted": "publico"},
            }
        )
        self.assertIn("texto:publico", text)
        self.assertNotIn("secreto", text)

    def test_structured_contraparte_folio_fecha_total(self):
        text = build_input_text(
            {
                "kind": "purchase",
                "source": {},
                "structured": {
                    "counterpartyName": "Proveedor SPA",
                    "documentNumber": "99",
                    "issueDate": "2026-01-02",
                    "totals": {"total": {"amount": "1000", "currency": "CLP"}},
                },
                "company": {"giro": "comercio"},
            }
        )
        self.assertIn("contraparte:Proveedor SPA", text)
        self.assertIn("folio:99", text)
        self.assertIn("fecha:2026-01-02", text)
        self.assertIn("total:1000 CLP", text)
        self.assertIn("giro:comercio", text)

    def test_bank_memo_y_fecha_mov(self):
        text = build_input_text(
            {
                "kind": "bank_statement_line",
                "structured": {
                    "bank": {"memo": "Transferencia sueldo", "postedDate": "2026-03-01"},
                },
            }
        )
        self.assertIn("memo_banco:Transferencia sueldo", text)
        self.assertIn("fecha_mov:2026-03-01", text)

    def test_map_kind_to_document_kind(self):
        self.assertEqual(map_kind_to_document_kind("purchase"), "PURCHASE")
        self.assertEqual(map_kind_to_document_kind("sale"), "SALE")
        self.assertEqual(map_kind_to_document_kind("fee"), "FEE")
        self.assertEqual(map_kind_to_document_kind("bank_statement_line"), "BANK_STATEMENT")
        self.assertEqual(map_kind_to_document_kind("otro"), "PURCHASE")


class SalvageJsonTests(unittest.TestCase):
    def test_salvage_campos_clave_desde_texto_roto(self):
        raw = (
            'blah {"category": "gasto_arriendo", "taxTreatment": "vat_affected", '
            '"primaryAccountName": "Arriendos", "confidence": 0.81, '
            '"alternativeAccountNames": ["Caja", "IVA crédito fiscal"]} trailing'
        )
        out = _salvage_json_object(raw)
        self.assertEqual(out["category"], "gasto_arriendo")
        self.assertEqual(out["taxTreatment"], "vat_affected")
        self.assertEqual(out["primaryAccountName"], "Arriendos")
        self.assertAlmostEqual(out["confidence"], 0.81)
        self.assertEqual(out["alternativeAccountNames"], ["Caja", "IVA crédito fiscal"])

    def test_salvage_vacio(self):
        self.assertEqual(_salvage_json_object(""), {})
        self.assertEqual(_salvage_json_object(None), {})  # type: ignore[arg-type]

    def test_parse_model_json_valido(self):
        parsed = _parse_model_json(
            '{"category":"compra","taxTreatment":"unknown","primaryAccountName":"Caja"}'
        )
        self.assertEqual(parsed["category"], "compra")
        self.assertEqual(parsed["primaryAccountName"], "Caja")

    def test_parse_model_json_invalido_usa_salvage(self):
        parsed = _parse_model_json(
            'modelo dijo: "primaryAccountCode": "5.1.01.001" y "category": "gasto"'
        )
        self.assertEqual(parsed.get("primaryAccountCode"), "5.1.01.001")
        self.assertEqual(parsed.get("category"), "gasto")


class ClassifyWithOllamaMockTests(unittest.TestCase):
    def test_run_classify_ok_con_ollama_mock(self):
        mock_embed = AsyncMock(return_value=[0.01] * 768)
        mock_chat = AsyncMock(
            return_value={
                "json": {
                    "category": "gasto_arriendo",
                    "taxTreatment": "vat_affected",
                    "primaryAccountName": "Arriendos",
                    "confidence": 0.88,
                    "alternativeAccountNames": ["Caja"],
                },
                "latencyMs": 15,
                "raw": "{}",
            }
        )

        with (
            patch("app.classify_engine.ollama_client.ollama_embed", mock_embed),
            patch("app.classify_engine.ollama_client.ollama_chat_json", mock_chat),
            patch("app.classify_engine.search_examples", return_value=[]),
        ):
            out = asyncio.run(run_classify(_classify_body()))

        self.assertEqual(out["json"]["outcome"], "suggested")
        self.assertEqual(out["json"]["provider"]["type"], "local")
        self.assertEqual(out["json"]["classification"]["category"], "gasto_arriendo")
        self.assertEqual(
            out["json"]["suggestedAccount"]["primary"]["accountId"],
            "acc-arriendo",
        )
        self.assertEqual(out["json"]["ragStatus"], "degraded")
        mock_embed.assert_awaited()
        mock_chat.assert_awaited()

    def test_run_classify_embedding_error_sin_llamar_chat(self):
        mock_embed = AsyncMock(side_effect=RuntimeError("ollama down"))
        mock_chat = AsyncMock()

        with (
            patch("app.classify_engine.ollama_client.ollama_embed", mock_embed),
            patch("app.classify_engine.ollama_client.ollama_chat_json", mock_chat),
        ):
            out = asyncio.run(run_classify(_classify_body()))

        self.assertEqual(out["json"]["outcome"], "error")
        self.assertEqual(out["json"]["ragStatus"], "failed")
        self.assertIn("Embedding error", out["json"]["errors"][0]["message"])
        mock_chat.assert_not_awaited()

    def test_run_classify_reintenta_si_json_incompleto(self):
        mock_embed = AsyncMock(return_value=[0.02] * 768)
        mock_chat = AsyncMock(
            side_effect=[
                {"json": {"category": "x"}, "latencyMs": 1, "raw": "{}"},
                {
                    "json": {
                        "category": "compra_general",
                        "taxTreatment": "unknown",
                        "primaryAccountName": "Caja",
                        "confidence": 0.6,
                    },
                    "latencyMs": 2,
                    "raw": "{}",
                },
            ]
        )

        with (
            patch("app.classify_engine.ollama_client.ollama_embed", mock_embed),
            patch("app.classify_engine.ollama_client.ollama_chat_json", mock_chat),
            patch("app.classify_engine.search_examples", return_value=[]),
        ):
            out = asyncio.run(run_classify(_classify_body()))

        self.assertEqual(out["json"]["outcome"], "suggested")
        self.assertEqual(mock_chat.await_count, 2)
        self.assertEqual(
            out["json"]["suggestedAccount"]["primary"]["accountId"],
            "acc-caja",
        )


if __name__ == "__main__":
    unittest.main()
