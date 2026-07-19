import unittest

from app.classify_engine import rank_examples_for_prompt


class RagCyclePromptTests(unittest.TestCase):
    """M01-023: al clasificar de nuevo, los ejemplos previos entran al ranking del prompt."""

    def test_segunda_clasificacion_prioriza_ejemplo_misma_contraparte(self):
        input_text = (
            "tipo:purchase\ncontraparte:acme spa\nfolio:1\nmonto:1000\ngiro:comercio"
        )
        examples = [
            {
                "id": "old",
                "inputText": "tipo:purchase\ncontraparte:otro ltda\nfolio:9",
                "dist": 0.10,
                "payloadJson": {"primaryAccountName": "Otro"},
            },
            {
                "id": "learned",
                "inputText": "tipo:purchase\ncontraparte:acme spa\nfolio:2",
                "dist": 0.25,
                "payloadJson": {"primaryAccountName": "Gastos generales"},
            },
        ]

        ranked = rank_examples_for_prompt(examples, input_text)
        self.assertEqual(ranked[0]["id"], "learned")
        self.assertEqual(
            ranked[0]["payloadJson"]["primaryAccountName"],
            "Gastos generales",
        )


if __name__ == "__main__":
    unittest.main()
