"""M01-028: lógica IVA chilena en prompt y post-proceso (19%, exento, activo fijo)."""

import unittest

from app.classify_engine import (
    CHILE_VAT_PROMPT_RULES,
    apply_chilean_vat_postprocess,
    build_system_prompt,
    looks_like_fixed_asset,
    resolve_vat_typical_rate,
)


CHART = [
    {"accountId": "a1", "code": "5.1.01", "name": "Gastos generales"},
    {
        "accountId": "a2",
        "code": "1.2.01.005",
        "name": "Equipos computacionales",
    },
    {
        "accountId": "a3",
        "code": "1.2.01.003",
        "name": "Maquinarias y equipos",
    },
    {"accountId": "a4", "code": "2.1.01.001", "name": "IVA crédito fiscal"},
]


def _purchase_inp(text: str, **extra) -> dict:
    inp: dict = {
        "kind": "purchase",
        "source": {"textRaw": text},
        "company": {"companyId": "c1", "giro": "comercio"},
        "period": {
            "companyId": "c1",
            "fiscalYear": 2026,
            "month": 3,
            "isClosed": False,
        },
        "accountingContext": {
            "chartOfAccountsTop": CHART,
            "rules": {"vatTypicalRate": 0.19},
        },
    }
    inp.update(extra)
    return inp


class ChileVatPromptTests(unittest.TestCase):
    def test_system_prompt_incluye_reglas_iva_chile(self):
        prompt = build_system_prompt(
            CHART, [], wants_entry=False, purpose="classify_purchase", vat_typical_rate=0.19
        )
        self.assertIn("REGLAS IVA CHILE", prompt)
        self.assertIn("19%", prompt)
        self.assertIn("ACTIVO FIJO", prompt)
        self.assertIn("vat_affected", prompt)
        self.assertIn("vat_exempt", prompt)
        self.assertIn(CHILE_VAT_PROMPT_RULES.split("\n")[0], prompt)
        self.assertIn("vatTypicalRate vigente", prompt)

    def test_resolve_vat_typical_rate_default_y_override(self):
        self.assertEqual(resolve_vat_typical_rate({}), 0.19)
        self.assertEqual(
            resolve_vat_typical_rate(
                {"accountingContext": {"rules": {"vatTypicalRate": 19}}}
            ),
            0.19,
        )
        self.assertEqual(
            resolve_vat_typical_rate(
                {"accountingContext": {"rules": {"vatTypicalRate": 0.19}}}
            ),
            0.19,
        )


class ChileVatPostprocessTests(unittest.TestCase):
    def test_compra_activo_fijo_sugiere_vat_affected(self):
        """Criterio de aceptación: compra activo fijo → tratamiento IVA correcto."""
        inp = _purchase_inp(
            "Factura compra notebook Dell para oficina. Neto 100000 IVA 19% Total 119000. "
            "Activo fijo / equipos computacionales."
        )
        cat, tax, primary, warnings = apply_chilean_vat_postprocess(
            category="expense",
            tax_treatment="unknown",
            primary_name="Gastos generales",
            inp=inp,
            purpose="classify_purchase",
            chart=CHART,
        )
        self.assertEqual(tax, "vat_affected")
        self.assertEqual(cat, "fixed_asset")
        self.assertIn("1.2", primary)
        self.assertTrue(any("activo fijo" in w.lower() or "fixed_asset" in w.lower() or "IVA" in w for w in warnings))

    def test_compra_activo_fijo_corrige_vat_exempt_erroneo(self):
        inp = _purchase_inp(
            "Compra maquinaria industrial factura afecta IVA 19% activo fijo"
        )
        cat, tax, primary, _ = apply_chilean_vat_postprocess(
            category="general",
            tax_treatment="vat_exempt",
            primary_name="5.1.01 - Gastos generales",
            inp=inp,
            purpose="classify_purchase",
            chart=CHART,
        )
        self.assertEqual(tax, "vat_affected")
        self.assertEqual(cat, "fixed_asset")
        self.assertTrue(
            primary.startswith("1.2") or "Maquinaria" in primary or "Equipos" in primary
        )

    def test_factura_exenta_queda_vat_exempt(self):
        inp = _purchase_inp("Factura exenta sin IVA — servicios educacionales")
        _, tax, _, _ = apply_chilean_vat_postprocess(
            category="expense",
            tax_treatment="vat_affected",
            primary_name="Gastos generales",
            inp=inp,
            purpose="classify_purchase",
            chart=CHART,
        )
        self.assertEqual(tax, "vat_exempt")

    def test_totales_con_iva_fuerzan_vat_affected(self):
        inp = _purchase_inp(
            "Compra insumos oficina",
            structured={
                "totals": {
                    "net": {"amount": "10000", "currency": "CLP"},
                    "tax": {"amount": "1900", "currency": "CLP"},
                    "total": {"amount": "11900", "currency": "CLP"},
                }
            },
        )
        _, tax, _, _ = apply_chilean_vat_postprocess(
            category="expense",
            tax_treatment="unknown",
            primary_name="Gastos generales",
            inp=inp,
            purpose="classify_purchase",
            chart=CHART,
        )
        self.assertEqual(tax, "vat_affected")

    def test_looks_like_fixed_asset_por_cuenta_1_2(self):
        self.assertTrue(
            looks_like_fixed_asset(
                text="factura",
                category="expense",
                primary_name="1.2.01.005 - Equipos computacionales",
            )
        )
        self.assertFalse(
            looks_like_fixed_asset(
                text="arriendo oficina mes marzo",
                category="rent",
                primary_name="Gastos de arriendo",
            )
        )


if __name__ == "__main__":
    unittest.main()
