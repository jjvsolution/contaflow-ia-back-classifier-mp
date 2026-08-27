from contextlib import contextmanager
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.config import settings

# Prisma acepta query params que libpq/psycopg rechazan (p. ej. ?schema=public).
_PRISMA_ONLY_QUERY_PARAMS = frozenset(
    {
        "schema",
        "connection_limit",
        "pool_timeout",
        "pgbouncer",
        "socket_timeout",
        "statement_cache_size",
    }
)
# Tablas Prisma viven en @@schema("contaflow"), no en public.
_DEFAULT_SEARCH_PATH = ("contaflow", "public")


def conninfo_from_database_url(database_url: str) -> tuple[str, tuple[str, ...]]:
    """Quita params de Prisma y deriva search_path para psycopg."""
    raw = database_url.strip().strip('"').strip("'")
    parsed = urlparse(raw)
    kept: list[tuple[str, str]] = []
    prisma_schema: str | None = None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered == "schema":
            prisma_schema = value.strip() or None
            continue
        if lowered in _PRISMA_ONLY_QUERY_PARAMS:
            continue
        kept.append((key, value))
    conninfo = urlunparse(parsed._replace(query=urlencode(kept)))
    search_path = list(_DEFAULT_SEARCH_PATH)
    if prisma_schema and prisma_schema not in search_path:
        search_path.insert(0, prisma_schema)
    return conninfo, tuple(search_path)


@contextmanager
def get_conn():
    conninfo, search_path = conninfo_from_database_url(settings.database_url)
    with psycopg.connect(conninfo) as conn:
        conn.execute(
            sql.SQL("SET search_path TO {}").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in search_path)
            )
        )
        yield conn


def postgres_ready_check() -> dict:
    """Ping de PostgreSQL para readiness (M01-020)."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"status": "up"}
    except Exception as e:
        return {"status": "down", "error": str(e)}


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


def find_example_id_by_source(
    tenant_id: str,
    scope: str,
    source_type: str,
    source_id: str,
) -> str | None:
    """Busca ejemplo RAG por clave lógica (M01-024)."""
    sql = """
    SELECT id FROM "AccountingKnowledgeExample"
    WHERE "tenantId" = %s
      AND "scope" = %s::"KnowledgeScope"
      AND "sourceType" = %s
      AND "sourceId" = %s
    ORDER BY "createdAt" DESC
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, scope, source_type, source_id))
            row = cur.fetchone()
            return str(row[0]) if row else None


def upsert_example(
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
) -> dict:
    """
    Inserta o actualiza un ejemplo RAG.
    Si hay sourceType+sourceId, el segundo learn actualiza (no duplica) — M01-024.
    """
    import uuid

    vec_lit = vector_literal(embedding)
    can_upsert = bool(source_type and source_id)

    if can_upsert:
        existing_id = find_example_id_by_source(
            tenant_id, scope, source_type, source_id
        )
        if existing_id:
            sql_update = """
            UPDATE "AccountingKnowledgeExample"
            SET "companyId" = %s,
                "giroKey" = %s,
                "documentKind" = %s::"DocumentKind",
                "inputText" = %s,
                "payloadJson" = %s::jsonb,
                "embedding" = %s::vector
            WHERE id = %s;
            """
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql_update,
                        (
                            company_id,
                            giro_key,
                            document_kind,
                            input_text,
                            Json(payload),
                            vec_lit,
                            existing_id,
                        ),
                    )
                conn.commit()
            return {"id": existing_id, "updated": True}

    row_id = str(uuid.uuid4())
    sql_insert = """
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
                sql_insert,
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
    return {"id": row_id, "updated": False}


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
    """Compat: delega en upsert_example y devuelve el id."""
    result = upsert_example(
        tenant_id=tenant_id,
        company_id=company_id,
        giro_key=giro_key,
        scope=scope,
        document_kind=document_kind,
        source_type=source_type,
        source_id=source_id,
        input_text=input_text,
        payload=payload,
        embedding=embedding,
    )
    return result["id"]
