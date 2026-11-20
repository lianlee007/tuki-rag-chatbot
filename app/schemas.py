"""Contratos públicos de la API del gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=8000)
    system: str | None = Field(default=None, max_length=4000)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=4096)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    attempts: int
    latency_ms: int
    request_id: str


class MensajeHistorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rol: Literal["user", "assistant"]
    contenido: str = Field(min_length=1, max_length=2000)


class RecomendarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mensaje: str = Field(min_length=2, max_length=1000)
    historial: list[MensajeHistorial] = Field(default_factory=list, max_length=8)
    ciudad: str | None = Field(default=None, max_length=80)
    restaurante_slug: str | None = Field(default=None, max_length=80)


class FuenteResponse(BaseModel):
    id: str
    tipo: Literal["plato", "restaurante"]
    nombre: str
    restaurante: str
    restaurante_slug: str
    ciudad: str | None = None
    categoria: str | None = None
    precio: float | None = None
    moneda: str = "COP"
    imagen_url: str | None = None
    vegetariano: bool = False
    vegano: bool = False
    picante: bool = False
    alergenos: list[str] = Field(default_factory=list)


class RecomendarResponse(BaseModel):
    respuesta: str
    fuentes: list[FuenteResponse]
    modelo: str
    usage: Usage
    attempts: int
    latency_ms: int
    recuperacion_ms: int
    contexto_tokens: int
    request_id: str


class ConocimientoResponse(BaseModel):
    documentos: int
    platos: int
    restaurantes: int
    embeddings: int
    ciudades: list[str]
    actualizado_en: str | None = None
    ttl_segundos: int
    fresco: bool
    fuente: str
    modelo_chat: str
    modelo_embeddings: str
    error: str | None = None
    error_embeddings: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
