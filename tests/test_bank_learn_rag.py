"""Aprendizaje RAG para cartolas confirmadas manualmente."""

from __future__ import annotations

import unittest

from app.classify_engine import (
    bank_suggested_entry_from_learn_payload,
    build_rag_short_circuit_result,
    pick_rag_short_circuit_example,
    primary_from_learn_payload,
)
from app.input_text import build_input_text, extract_bank_counterparty


class BankLearnInputTextTests(unittest.TestCase):
    def test_extrae_contraparte_desde_glosa_pago(self):
        self.assertEqual(
            extract_bank_counterparty("Pago Municipalidad Demo SpA folio 10006"),
            "Municipalidad Demo SpA",
        )

    def test_build_input_text_incluye_contraparte_en_cartola(self):
        text = build_input_text(
            {
                "kind": "bank_statement_line",
                "source": {"textRaw": "Pago Higiene Demo SpA"},
                "structured": {
                    "bank": {"memo": "Pago Higiene Demo SpA"},
                    "totals": {"total": {"amount": "-1000", "currency": "CLP"}},
                },
            }
        )
        self.assertIn("contraparte:Higiene Demo SpA", text)


class BankLearnPayloadTests(unittest.TestCase):
    def test_primary_from_payload_cartola(self):
        payload = {
            "primaryAccountName": "2.1.01.001 - Proveedores nacionales",
            "bankAccountName": "1.1.01.003 - Banco cuenta corriente",
        }
        self.assertIn("Proveedores", primary_from_learn_payload(payload))

    def test_short_circuit_acepta_payload_cartola(self):
        ex = pick_rag_short_circuit_example(
            [
                {
                    "id": "bank-1",
                    "dist": 0.10,
                    "inputText": (
                        "tipo:bank_statement_line\n"
                        "contraparte:municipalidad demo spa\n"
                        "texto:Pago Municipalidad Demo SpA"
                    ),
                    "payloadJson": {
                        "primaryAccountName": "2.1.01.001 - Proveedores nacionales",
                        "bankAccountName": "1.1.01.003 - Banco cuenta corriente",
                        "classification": {
                            "category": "proveedores_nacionales",
                            "taxTreatment": "unknown",
                        },
                    },
                }
            ],
            (
                "tipo:bank_statement_line\n"
                "contraparte:municipalidad demo spa\n"
                "texto:Pago Municipalidad Demo SpA folio 10007"
            ),
            max_dist=0.22,
            max_dist_same_counterparty=0.35,
        )
        self.assertIsNotNone(ex)
        self.assertEqual(ex["id"], "bank-1")

    def test_build_result_incluye_asiento_banco_contrapartida(self):
        chart = [
            {
                "accountId": "acc-banco",
                "code": "1.1.01.003",
                "name": "Banco cuenta corriente",
            },
            {
                "accountId": "acc-prov",
                "code": "2.1.01.001",
                "name": "Proveedores nacionales",
            },
        ]
        inp = {
            "kind": "bank_statement_line",
            "source": {"textRaw": "Pago Municipalidad Demo SpA"},
            "structured": {
                "bank": {
                    "memo": "Pago Municipalidad Demo SpA",
                    "postedDate": "2026-01-15",
                },
                "totals": {"total": {"amount": "-50000", "currency": "CLP"}},
            },
        }
        out = build_rag_short_circuit_result(
            body={"requestId": "r-bank"},
            request_id="r-bank",
            inp=inp,
            kind="bank_statement_line",
            purpose="classify_bank_line",
            example={
                "id": "bank-1",
                "dist": 0.08,
                "payloadJson": {
                    "primaryAccountId": "acc-prov",
                    "primaryAccountName": "2.1.01.001 - Proveedores nacionales",
                    "bankAccountId": "acc-banco",
                    "bankAccountName": "1.1.01.003 - Banco cuenta corriente",
                    "classification": {
                        "category": "proveedores_nacionales",
                        "taxTreatment": "unknown",
                    },
                },
            },
            examples=[{"id": "bank-1"}],
            chart=chart,
            input_text=build_input_text(inp),
            wants_entry=True,
            explain=True,
            t0=__import__("time").perf_counter(),
        )
        result = out["json"]
        entry = result["suggestedEntry"]["entry"]
        lines = entry["lines"]
        self.assertEqual(lines[0]["account"]["accountId"], "acc-banco")
        self.assertEqual(lines[1]["account"]["accountId"], "acc-prov")
        self.assertEqual(result["suggestedAccount"]["primary"]["accountId"], "acc-prov")

    def test_bank_suggested_entry_from_payload(self):
        chart = [
            {"accountId": "a1", "code": "1.1.01.001", "name": "Caja"},
            {"accountId": "a2", "code": "5.1.01.001", "name": "Gastos"},
        ]
        suggested = bank_suggested_entry_from_learn_payload(
            {
                "bankAccountName": "1.1.01.001 - Caja",
                "counterAccountName": "5.1.01.001 - Gastos",
            },
            {
                "structured": {
                    "totals": {"total": {"amount": "-1000"}},
                    "bank": {"memo": "Pago demo"},
                }
            },
            chart,
        )
        self.assertIsNotNone(suggested)
        lines = suggested["entry"]["lines"]
        self.assertEqual(lines[0]["account"]["accountId"], "a1")
        self.assertEqual(lines[1]["account"]["accountId"], "a2")


if __name__ == "__main__":
    unittest.main()
