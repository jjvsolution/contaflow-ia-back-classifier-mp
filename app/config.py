from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Puede ser la misma URL de Prisma (?schema=contaflow). db.py la sanitiza para psycopg.
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/contaflow",
        validation_alias="DATABASE_URL",
    )
    ollama_host: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_HOST")
    ollama_chat_model: str = Field(default="llama3.2", validation_alias="OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", validation_alias="OLLAMA_EMBED_MODEL")
    embedding_dimensions: int = Field(default=768, validation_alias="EMBEDDING_DIMENSIONS")
    internal_token: str | None = Field(default=None, validation_alias="INTERNAL_TOKEN")
    rag_company_limit: int = Field(default=8, validation_alias="RAG_COMPANY_LIMIT")
    rag_giro_limit: int = Field(default=8, validation_alias="RAG_GIRO_LIMIT")
    rag_short_circuit_enabled: bool = Field(
        default=True, validation_alias="RAG_SHORT_CIRCUIT_ENABLED"
    )
    # Distancia coseno pgvector (<=>): menor = más parecido. 0 = idéntico.
    rag_short_circuit_max_dist: float = Field(
        default=0.22, validation_alias="RAG_SHORT_CIRCUIT_MAX_DIST"
    )
    rag_short_circuit_max_dist_same_cp: float = Field(
        default=0.35, validation_alias="RAG_SHORT_CIRCUIT_MAX_DIST_SAME_CP"
    )


settings = Settings()
