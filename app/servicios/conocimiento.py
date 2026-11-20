"""Mantiene el índice del catálogo en memoria."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from app.config import Settings
from app.dominio.conocimiento import Documento
from app.dominio.errores import ErrorIA
from app.dominio.recuperacion import IndiceRAG

logger = logging.getLogger(__name__)


class ExtractorConocimiento(Protocol):
    async def extraer(self) -> list[Documento]: ...


class ProveedorEmbeddings(Protocol):
    async def embed(self, textos: list[str]) -> list[list[float]]: ...


class ServicioConocimiento:
    def __init__(
        self,
        extractor: ExtractorConocimiento,
        embeddings: ProveedorEmbeddings,
        settings: Settings,
    ) -> None:
        self._extractor = extractor
        self._embeddings = embeddings
        self._settings = settings
        self._lock = asyncio.Lock()
        self._indice: IndiceRAG | None = None
        self._actualizado: float = 0.0
        self._error: str | None = None
        self._error_embeddings: str | None = None
        self._firma: str | None = None
        self._vectores: list[list[float]] = []

    async def obtener(self) -> IndiceRAG:
        if self._indice is not None and not self._esta_vencido():
            return self._indice
        async with self._lock:
            if self._indice is not None and not self._esta_vencido():
                return self._indice
            try:
                await self._reconstruir()
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)[:300]
                if self._indice is not None:
                    logger.warning(
                        "conocimiento_refresco_fallido",
                        extra={"event": "knowledge_stale", "error": self._error},
                    )
                    return self._indice
                raise
        assert self._indice is not None
        return self._indice

    async def recargar(self) -> dict[str, Any]:
        async with self._lock:
            await self._reconstruir()
        return self.estado()

    async def _reconstruir(self) -> None:
        documentos = await self._extractor.extraer()
        if not documentos:
            raise ErrorIA(
                "El catálogo no tiene platos disponibles para recomendar.",
                status_code=503,
                code="knowledge_empty",
            )
        firma = self._firma_documentos(documentos)
        reutilizados = firma == self._firma and len(self._vectores) == len(documentos)
        vectores = self._vectores if reutilizados else []
        self._error_embeddings = None
        if not reutilizados:
            try:
                vectores = await self._embeddings.embed(
                    [documento.texto for documento in documentos]
                )
                if vectores:
                    self._firma = firma
                    self._vectores = vectores
            except Exception as exc:  # noqa: BLE001
                self._error_embeddings = str(exc)[:300]
                logger.warning(
                    "embeddings_no_disponibles",
                    extra={"event": "embeddings_error", "error": self._error_embeddings},
                )

        self._indice = IndiceRAG(documentos, vectores)
        self._actualizado = time.time()
        self._error = None
        logger.info(
            "conocimiento_actualizado",
            extra={
                "event": "knowledge_refresh",
                "documentos": len(documentos),
                "embeddings": len(vectores),
                "embeddings_reutilizados": reutilizados,
            },
        )

    @staticmethod
    def _firma_documentos(documentos: list[Documento]) -> str:
        hasher = hashlib.sha256()
        for documento in documentos:
            hasher.update(documento.id.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(documento.texto.encode("utf-8"))
            hasher.update(b"\x1f")
        return hasher.hexdigest()

    def _esta_vencido(self) -> bool:
        return (time.time() - self._actualizado) > self._settings.conocimiento_ttl_seconds

    def estado(self) -> dict[str, Any]:
        estadisticas = (
            self._indice.estadisticas()
            if self._indice is not None
            else {
                "documentos": 0,
                "platos": 0,
                "restaurantes": 0,
                "embeddings": 0,
                "ciudades": [],
            }
        )
        return {
            **estadisticas,
            "actualizado_en": (
                datetime.fromtimestamp(self._actualizado, tz=UTC).isoformat()
                if self._actualizado
                else None
            ),
            "ttl_segundos": self._settings.conocimiento_ttl_seconds,
            "fresco": bool(self._indice is not None and not self._esta_vencido()),
            "fuente": "catalogo",
            "modelo_chat": self._settings.modelo_chat,
            "modelo_embeddings": self._settings.modelo_embeddings,
            "error": self._error,
            "error_embeddings": self._error_embeddings,
        }
