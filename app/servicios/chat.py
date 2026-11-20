"""Caso de uso de generación de texto.

Depende solo de puertos (``ProveedorIA``), nunca de Groq ni de FastAPI. Ese es
el punto de la arquitectura hexagonal: cambiar de proveedor no toca esta capa.
"""

from __future__ import annotations

from app.dominio.errores import PromptDemasiadoLargo
from app.dominio.modelos import ResultadoIA
from app.puertos import ProveedorIA


class ServicioChat:
    def __init__(self, proveedor: ProveedorIA, *, max_chars: int) -> None:
        self._proveedor = proveedor
        self._max_chars = max_chars

    async def generar(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        request_id: str,
    ) -> ResultadoIA:
        if len(prompt) > self._max_chars:
            raise PromptDemasiadoLargo(self._max_chars)
        return await self._proveedor.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        )
