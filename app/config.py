"""Configuración del chatbot RAG local.

Todo se puede cambiar con variables de entorno. El catálogo vive en un JSON
exportado de PostgreSQL para que el repositorio sea autocontenido.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "tuki-rag-chatbot"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Ollama -------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_modelo_chat: str = "tuki-chef"
    ollama_modelo_respaldo: str = "qwen2.5:1.5b-instruct"
    ollama_modelo_embeddings: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 120.0
    ollama_total_timeout_seconds: float = 180.0
    ollama_max_attempts: int = 2
    ollama_backoff_seconds: float = 0.5
    ollama_max_backoff_seconds: float = 4.0
    ollama_keep_alive: str = "15m"
    ollama_num_ctx: int = 3072

    # --- Catálogo y RAG -----------------------------------------------------
    catalogo_path: str = "datos/catalogo.json"
    conocimiento_ttl_seconds: int = 300
    rag_chunk_tokens: int = 220
    rag_chunk_overlap: int = 40
    rag_candidatos: int = 20
    rag_top_k: int = 6
    rag_contexto_max_tokens: int = 1100
    rag_mmr_lambda: float = 0.72
    rag_umbral_semantico: float = 0.22

    # --- API ----------------------------------------------------------------
    request_max_chars: int = 4000
    max_output_tokens: int = 1024

    @field_validator("ollama_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def modelo_chat(self) -> str:
        return self.ollama_modelo_chat.strip()

    @property
    def modelo_embeddings(self) -> str:
        return self.ollama_modelo_embeddings.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return ["*"]

    def validate_or_raise(self) -> None:
        errors: list[str] = []
        for name in (
            "conocimiento_ttl_seconds",
            "rag_chunk_tokens",
            "rag_candidatos",
            "rag_top_k",
            "rag_contexto_max_tokens",
            "request_max_chars",
            "max_output_tokens",
        ):
            if getattr(self, name) < 1:
                errors.append(f"{name} debe ser mayor que cero")
        if self.rag_chunk_overlap < 0:
            errors.append("rag_chunk_overlap no puede ser negativo")
        if self.rag_chunk_overlap >= self.rag_chunk_tokens:
            errors.append("rag_chunk_overlap debe ser menor que rag_chunk_tokens")
        if not 0.0 <= self.rag_mmr_lambda <= 1.0:
            errors.append("rag_mmr_lambda debe estar entre 0 y 1")
        if not self.modelo_chat:
            errors.append("OLLAMA_MODELO_CHAT no puede estar vacío")
        if not self.modelo_embeddings:
            errors.append("OLLAMA_MODELO_EMBEDDINGS no puede estar vacío")
        if not self.ollama_url:
            errors.append("OLLAMA_URL no puede estar vacío")
        if not self.catalogo_path:
            errors.append("CATALOGO_PATH no puede estar vacío")
        if errors:
            raise RuntimeError("Configuración inválida: " + "; ".join(errors))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
