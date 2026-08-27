"""URL Prisma (?schema=) usable por psycopg."""

import unittest

from app.db import conninfo_from_database_url


class PrismaDatabaseUrlTests(unittest.TestCase):
    def test_quita_schema_y_conserva_sslmode(self):
        conninfo, search_path = conninfo_from_database_url(
            "postgresql://postgres:postgres@localhost:5432/contaflow"
            "?schema=public&sslmode=require"
        )
        self.assertNotIn("schema", conninfo)
        self.assertIn("sslmode=require", conninfo)
        self.assertEqual(search_path, ("contaflow", "public"))

    def test_schema_contaflow_queda_en_search_path(self):
        conninfo, search_path = conninfo_from_database_url(
            "postgresql://u:p@db:5432/contaflow?schema=contaflow"
        )
        self.assertEqual(
            conninfo, "postgresql://u:p@db:5432/contaflow"
        )
        self.assertEqual(search_path, ("contaflow", "public"))

    def test_schema_custom_va_primero(self):
        _, search_path = conninfo_from_database_url(
            "postgresql://u:p@db:5432/x?schema=otro"
        )
        self.assertEqual(search_path, ("otro", "contaflow", "public"))

    def test_quita_connection_limit_de_prisma(self):
        conninfo, _ = conninfo_from_database_url(
            "postgresql://u:p@db:5432/x?connection_limit=5&schema=public"
        )
        self.assertNotIn("connection_limit", conninfo)
        self.assertNotIn("schema", conninfo)

    def test_url_sin_query_queda_igual(self):
        url = "postgresql://postgres:postgres@localhost:5432/contaflow"
        conninfo, search_path = conninfo_from_database_url(url)
        self.assertEqual(conninfo, url)
        self.assertEqual(search_path, ("contaflow", "public"))

    def test_acepta_url_entrecomillada(self):
        conninfo, _ = conninfo_from_database_url(
            '"postgresql://u:p@db:5432/x?schema=public"'
        )
        self.assertEqual(conninfo, "postgresql://u:p@db:5432/x")
