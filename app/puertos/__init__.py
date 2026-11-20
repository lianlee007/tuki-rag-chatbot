"""Contratos que los adaptadores implementan."""

from __future__ import annotations

from typing import Protocol

from app.dominio.modelos import ResultadoIA


class ProveedorIA(Protocol):
    async def generate(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        request_id: str,
    ) -> ResultadoIA: ...


class ProveedorEmbeddings(Protocol):
    async def embed(self, textos: list[str]) -> list[list[float]]: ...
