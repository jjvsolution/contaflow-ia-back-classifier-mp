from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.config import settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url) as conn:
        yield conn


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


def search_examples(
    tenant_id: str,
    company_id: str,
    giro_key: str,
    document_kind: str,
    query_embedding: list[float],
    company_limit: int,
    giro_limit: int,
) -> list[dict]:
    vec_lit = vector_literal(query_embedding)
    results: list[dict] = []
    seen: set[str] = set()

    sql_company = """
    SELECT id, "inputText", "payloadJson", "scope",
           ("embedding" <=> %s::vector) AS dist
    FROM "AccountingKnowledgeExample"
    WHERE "tenantId" = %s
      AND "companyId" = %s
      AND "documentKind" = %s::"DocumentKind"
      AND "embedding" IS NOT NULL
    ORDER BY "embedding" <=> %s::vector
    LIMIT %s;
    """

    sql_giro = """
    SELECT id, "inputText", "payloadJson", "scope",
           ("embedding" <=> %s::vector) AS dist
    FROM "AccountingKnowledgeExample"
    WHERE "tenantId" = %s
      AND "scope" = 'GIRO'::"KnowledgeScope"
      AND "giroKey" = %s
      AND "companyId" IS NULL
      AND "documentKind" = %s::"DocumentKind"
      AND "embedding" IS NOT NULL
    ORDER BY "embedding" <=> %s::vector
    LIMIT %s;
    """

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql_company,
                (vec_lit, tenant_id, company_id, document_kind, vec_lit, company_limit),
            )
            for row in cur.fetchall():
                rid = row["id"]
                if rid not in seen:
                    seen.add(rid)
                    results.append(dict(row))

            cur.execute(
                sql_giro,
                (vec_lit, tenant_id, giro_key, document_kind, vec_lit, giro_limit),
            )
            for row in cur.fetchall():
                rid = row["id"]
                if rid not in seen:
                    seen.add(rid)
                    results.append(dict(row))

    results.sort(key=lambda r: float(r["dist"]))
    return results


def insert_example(
    tenant_id: str,
    company_id: str | None,
    giro_key: str,
    scope: str,
    document_kind: str,
    source_type: str | None,
    source_id: str | None,
    input_text: str,
    payload: dict,
    embedding: list[float],
) -> str:
    import uuid

    row_id = str(uuid.uuid4())
    vec_lit = vector_literal(embedding)

    sql = """
    INSERT INTO "AccountingKnowledgeExample"
      (id, "tenantId", "companyId", "giroKey", "scope", "documentKind",
       "sourceType", "sourceId", "inputText", "payloadJson", "embedding")
    VALUES
      (%s, %s, %s, %s, %s::"KnowledgeScope", %s::"DocumentKind",
       %s, %s, %s, %s::jsonb, %s::vector);
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    row_id,
                    tenant_id,
                    company_id,
                    giro_key,
                    scope,
                    document_kind,
                    source_type,
                    source_id,
                    input_text,
                    Json(payload),
                    vec_lit,
                ),
            )
        conn.commit()
    return row_id
