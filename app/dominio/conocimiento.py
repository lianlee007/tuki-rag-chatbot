"""Modelos del conocimiento gastronómico que alimenta el RAG.

Un ``Documento`` es la unidad que se indexa y se cita. Los platos son atómicos
para que el precio nunca se separe de su descripción; los restaurantes pueden
partirse en varios fragmentos si su descripción es larga.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TipoDocumento = Literal["plato", "restaurante"]


@dataclass(frozen=True, slots=True)
class Documento:
    id: str
    tipo: TipoDocumento
    texto: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fuente:
    id: str
    tipo: TipoDocumento
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
    alergenos: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Recuperado:
    documento: Documento
    puntaje: float = 0.0
    origen: str = "hibrido"


@dataclass(frozen=True, slots=True)
class ResultadoRecomendacion:
    content: str
    model: str
    fuentes: tuple[Fuente, ...]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: int
    latency_ms: int
    recuperacion_ms: int
    contexto_tokens: int


def documento_a_dict(documento: Documento) -> dict[str, Any]:
    return {
        "id": documento.id,
        "tipo": documento.tipo,
        "texto": documento.texto,
        "metadata": documento.metadata,
    }


def documento_desde_dict(datos: dict[str, Any]) -> Documento:
    return Documento(
        id=str(datos["id"]),
        tipo=datos["tipo"],
        texto=str(datos["texto"]),
        metadata=dict(datos.get("metadata") or {}),
    )
