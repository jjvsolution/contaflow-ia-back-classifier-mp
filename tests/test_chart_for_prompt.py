"""Filtro de plan de cuentas en prompt por tipo de documento."""

import unittest

from app.classify_engine import CHART_PROMPT_LIMIT, chart_for_prompt

SAMPLE_CHART = [
    {"code": "1.1.01.001", "name": "Caja", "type": "ASSET"},
    {"code": "1.2.01.001", "name": "IVA crédito", "type": "ASSET"},
    {"code": "2.1.01.001", "name": "Proveedores", "type": "LIABILITY"},
    {"code": "4.1.01.001", "name": "Ventas", "type": "INCOME"},
    {"code": "5.1.01.001", "name": "Arriendos", "type": "EXPENSE"},
    {"code": "5.1.02.001", "name": "Honorarios", "type": "EXPENSE"},
]


class ChartForPromptTests(unittest.TestCase):
    def test_purchase_solo_gastos_y_activos(self):
        out = chart_for_prompt(SAMPLE_CHART, "purchase")
        codes = {a["code"] for a in out}
        self.assertIn("1.1.01.001", codes)
        self.assertIn("1.2.01.001", codes)
        self.assertIn("5.1.01.001", codes)
        self.assertNotIn("2.1.01.001", codes)
        self.assertNotIn("4.1.01.001", codes)

    def test_sale_solo_ingresos(self):
        out = chart_for_prompt(SAMPLE_CHART, "sale")
        self.assertEqual([a["code"] for a in out], ["4.1.01.001"])

    def test_fee_solo_gastos(self):
        out = chart_for_prompt(SAMPLE_CHART, "fee")
        codes = {a["code"] for a in out}
        self.assertIn("5.1.01.001", codes)
        self.assertIn("5.1.02.001", codes)
        self.assertNotIn("1.1.01.001", codes)
        self.assertNotIn("2.1.01.001", codes)

    def test_limite_30_cuentas(self):
        big = [
            {"code": f"5.1.{i:02d}.001", "name": f"Gasto {i}", "type": "EXPENSE"}
            for i in range(40)
        ]
        out = chart_for_prompt(big, "fee", limit=CHART_PROMPT_LIMIT)
        self.assertEqual(len(out), CHART_PROMPT_LIMIT)

    def test_match_por_type_sin_codigo_estandar(self):
        chart = [
            {"code": "X.9.001", "name": "Ingreso atípico", "type": "INCOME"},
            {"code": "5.1.01.001", "name": "Gasto", "type": "EXPENSE"},
        ]
        out = chart_for_prompt(chart, "sale")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Ingreso atípico")
