"""Atajo RAG: clasificar sin LLM cuando hay ejemplo histórico muy similar."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.classify_engine import (
    build_rag_short_circuit_result,
    pick_rag_short_circuit_example,
    run_classify,
)
from tests.test_m24_006_helpers_classify import CHART, _classify_body


class RagShortCircuitPickTests(unittest.TestCase):
    def test_acepta_ejemplo_cercano(self):
        ex = pick_rag_short_circuit_example(
            [
                {
                    "id": "1",
                    "dist": 0.12,
                    "inputText": "tipo:purchase\ncontraparte:energia demo spa",
                    "payloadJson": {"primaryAccountName": "Electricidad"},
                }
            ],
            "tipo:purchase\ncontraparte:energia demo spa\nfolio:10099",
            max_dist=0.22,
            max_dist_same_counterparty=0.35,
        )
        self.assertIsNotNone(ex)
        self.assertEqual(ex["id"], "1")

    def test_rechaza_si_distancia_alta_sin_misma_contraparte(self):
        ex = pick_rag_short_circuit_example(
            [
                {
                    "id": "1",
                    "dist": 0.40,
                    "inputText": "tipo:purchase\ncontraparte:otro",
                    "payloadJson": {"primaryAccountName": "Gastos"},
                }
            ],
            "tipo:purchase\ncontraparte:energia demo spa",
            max_dist=0.22,
            max_dist_same_counterparty=0.35,
        )
        self.assertIsNone(ex)

    def test_acepta_distancia_media_si_misma_contraparte(self):
        ex = pick_rag_short_circuit_example(
            [
                {
                    "id": "1",
                    "dist": 0.30,
                    "inputText": "tipo:purchase\ncontraparte:energia demo spa",
                    "payloadJson": {"primaryAccountName": "Electricidad"},
                }
            ],
            "tipo:purchase\ncontraparte:energia demo spa\nfolio:10100",
            max_dist=0.22,
            max_dist_same_counterparty=0.35,
        )
        self.assertIsNotNone(ex)


class RagShortCircuitClassifyTests(unittest.TestCase):
    def test_run_classify_no_llama_chat_si_hay_atajo(self):
        body = _classify_body()
        examples = [
            {
                "id": "learned-1",
                "dist": 0.08,
                "inputText": (
                    "tipo:purchase\n"
                    "contraparte:Inmobiliaria Demo\n"
                    "texto:Factura de arriendo"
                ),
                "payloadJson": {
                    "primaryAccountName": "Arriendos",
                    "classification": {
                        "category": "gasto_arriendo",
                        "taxTreatment": "vat_affected",
                    },
                },
            }
        ]

        with (
            patch(
                "app.classify_engine.ollama_client.ollama_embed",
                new=AsyncMock(return_value=[0.1] * 768),
            ),
            patch(
                "app.classify_engine.search_examples",
                return_value=examples,
            ),
            patch(
                "app.classify_engine.ollama_client.ollama_chat_json",
                new=AsyncMock(),
            ) as chat_mock,
        ):
            out = asyncio.run(run_classify(body))

        chat_mock.assert_not_called()
        result = out["json"]
        self.assertEqual(result["outcome"], "suggested")
        self.assertEqual(result["provider"]["model"], "rag-short-circuit")
        self.assertEqual(result["suggestedAccount"]["primary"]["name"], "Arriendos")
        self.assertTrue(
            any("sin LLM" in w for w in (result.get("warnings") or []))
        )

    def test_build_result_incluye_cuenta_del_payload(self):
        out = build_rag_short_circuit_result(
            body={"requestId": "r1"},
            request_id="r1",
            inp=_classify_body()["input"],
            kind="purchase",
            purpose="classify_purchase",
            example={
                "id": "x",
                "dist": 0.1,
                "payloadJson": {
                    "primaryAccountId": "acc-arriendo",
                    "primaryAccountName": "Arriendos",
                    "classification": {
                        "category": "gasto_arriendo",
                        "taxTreatment": "vat_affected",
                    },
                },
            },
            examples=[{"id": "x"}],
            chart=CHART,
            input_text="tipo:purchase",
            wants_entry=False,
            explain=True,
            t0=__import__("time").perf_counter(),
        )
        primary = out["json"]["suggestedAccount"]["primary"]
        self.assertEqual(primary["accountId"], "acc-arriendo")
