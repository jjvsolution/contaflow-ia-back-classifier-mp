"""M01-024: segundo learn del mismo sourceId actualiza, no duplica."""

import unittest
from unittest.mock import MagicMock, patch

from app.db import find_example_id_by_source, upsert_example


class UpsertRagSourceTests(unittest.TestCase):
    def test_find_example_id_by_source_devuelve_id(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("existing-id",)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("app.db.get_conn") as get_conn:
            get_conn.return_value.__enter__.return_value = mock_conn
            found = find_example_id_by_source(
                "t1", "COMPANY", "document", "doc-1"
            )

        self.assertEqual(found, "existing-id")
        mock_cur.execute.assert_called_once()
        args = mock_cur.execute.call_args[0]
        self.assertIn("sourceId", args[0])
        self.assertEqual(args[1], ("t1", "COMPANY", "document", "doc-1"))

    def test_segundo_learn_actualiza_mismo_id(self):
        mock_cur = MagicMock()
        # find → existing; update → ok
        mock_cur.fetchone.return_value = ("same-id",)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("app.db.get_conn") as get_conn:
            get_conn.return_value.__enter__.return_value = mock_conn
            result = upsert_example(
                tenant_id="t1",
                company_id="c1",
                giro_key="comercio",
                scope="COMPANY",
                document_kind="PURCHASE",
                source_type="document",
                source_id="doc-1",
                input_text="texto actualizado",
                payload={"primaryAccountName": "Gastos"},
                embedding=[0.1, 0.2],
            )

        self.assertEqual(result["id"], "same-id")
        self.assertTrue(result["updated"])
        sqls = [c.args[0] for c in mock_cur.execute.call_args_list]
        self.assertTrue(any("UPDATE" in s for s in sqls))
        self.assertFalse(any("INSERT" in s for s in sqls))

    def test_primer_learn_inserta(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("app.db.get_conn") as get_conn:
            get_conn.return_value.__enter__.return_value = mock_conn
            result = upsert_example(
                tenant_id="t1",
                company_id="c1",
                giro_key="comercio",
                scope="COMPANY",
                document_kind="PURCHASE",
                source_type="document",
                source_id="doc-nuevo",
                input_text="texto nuevo",
                payload={"primaryAccountName": "Caja"},
                embedding=[0.3],
            )

        self.assertFalse(result["updated"])
        self.assertTrue(result["id"])
        sqls = [c.args[0] for c in mock_cur.execute.call_args_list]
        self.assertTrue(any("INSERT" in s for s in sqls))
        self.assertFalse(any(s.strip().startswith("UPDATE") for s in sqls))

    def test_sin_source_id_siempre_inserta(self):
        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("app.db.get_conn") as get_conn:
            get_conn.return_value.__enter__.return_value = mock_conn
            result = upsert_example(
                tenant_id="t1",
                company_id=None,
                giro_key="comercio",
                scope="GIRO",
                document_kind="PURCHASE",
                source_type=None,
                source_id=None,
                input_text="sin source",
                payload={},
                embedding=[0.0],
            )

        self.assertFalse(result["updated"])
        sqls = [c.args[0] for c in mock_cur.execute.call_args_list]
        self.assertEqual(len(sqls), 1)
        self.assertIn("INSERT", sqls[0])


if __name__ == "__main__":
    unittest.main()
