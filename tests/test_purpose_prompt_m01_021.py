import unittest

from app.classify_engine import (
    PURPOSE_PROMPT_VARIANTS,
    build_system_prompt,
    resolve_purpose,
)


class PurposePromptTests(unittest.TestCase):
    def test_cada_purpose_tiene_variante_distinta(self):
        required = {
            "classify_purchase",
            "classify_sale",
            "classify_fee",
            "classify_bank_line",
            "suggest_journal_entry",
        }
        self.assertTrue(required.issubset(PURPOSE_PROMPT_VARIANTS.keys()))

        texts = {
            p: build_system_prompt([], [], wants_entry=False, purpose=p)
            for p in required
        }
        # Cada variante debe aparecer en su prompt.
        self.assertIn("PURPOSE=classify_purchase", texts["classify_purchase"])
        self.assertIn("COMPRA", texts["classify_purchase"])
        self.assertIn("PURPOSE=classify_sale", texts["classify_sale"])
        self.assertIn("VENTA", texts["classify_sale"])
        self.assertIn("PURPOSE=classify_fee", texts["classify_fee"])
        self.assertIn("HONORARIOS", texts["classify_fee"])
        self.assertIn("PURPOSE=classify_bank_line", texts["classify_bank_line"])
        self.assertIn("CARTOLA", texts["classify_bank_line"])
        self.assertIn("PURPOSE=suggest_journal_entry", texts["suggest_journal_entry"])
        self.assertIn("journalLines", texts["suggest_journal_entry"])

        # No deben ser el mismo texto genérico.
        unique = set(texts.values())
        self.assertEqual(len(unique), len(required))

    def test_suggest_journal_entry_fuerza_asiento_aunque_wants_entry_false(self):
        prompt = build_system_prompt(
            [], [], wants_entry=False, purpose="suggest_journal_entry"
        )
        self.assertIn("Incluye journalLines", prompt)
        self.assertNotIn("NO incluyas journalLines", prompt)

    def test_resolve_purpose_usa_body_o_fallback_kind(self):
        self.assertEqual(
            resolve_purpose({"purpose": "classify_sale"}, "purchase"),
            "classify_sale",
        )
        self.assertEqual(resolve_purpose({}, "fee"), "classify_fee")
        self.assertEqual(
            resolve_purpose({}, "bank_statement_line"),
            "classify_bank_line",
        )


if __name__ == "__main__":
    unittest.main()
