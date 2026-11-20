"""Adaptador de Ollama: chat y embeddings en la misma máquina.

Se usa la API nativa (``/api/chat`` y ``/api/embed``) en lugar de la compatible
con OpenAI porque permite fijar ``keep_alive``, ``num_ctx`` y ``num_predict``
sin trucos. El modelo de chat es el personalizado ``tuki-chef``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx

from app.config import Settings
from app.dominio.errores import ErrorIA
from app.dominio.modelos import ResultadoIA
from app.dominio.texto import limpiar_salida

logger = logging.getLogger(__name__)

_LOTE_EMBEDDINGS = 24


class OllamaError(ErrorIA):
    """Error devuelto por el servidor local de Ollama."""


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self._url_chat = f"{settings.ollama_url}/api/chat"
        self._url_embed = f"{settings.ollama_url}/api/embed"

    async def generate(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        request_id: str,
    ) -> ResultadoIA:
        if not self._settings.modelo_chat:
            raise OllamaError(
                "El modelo local no está configurado.",
                status_code=503,
                code="ai_not_configured",
            )

        max_tokens = min(max_tokens, self._settings.max_output_tokens)
        started = time.perf_counter()
        deadline = started + self._settings.ollama_total_timeout_seconds
        ultimo_error = "La IA local no está disponible en este momento."
        agoto_tiempo = False
        modelo = self._settings.modelo_chat

        # Dos intentos: el segundo sirve para caer al modelo base si el
        # personalizado todavía no existe en Ollama.
        for intento in range(1, self._settings.ollama_max_attempts + 1):
            restante = deadline - time.monotonic()
            if restante <= 0.5:
                agoto_tiempo = True
                ultimo_error = "La IA local tardó demasiado en responder."
                break
            try:
                respuesta = await self._client.post(
                    self._url_chat,
                    json={
                        "model": modelo,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "keep_alive": self._settings.ollama_keep_alive,
                        "options": {
                            "temperature": max(0.0, min(2.0, temperature)),
                            "num_predict": max_tokens,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                            "num_ctx": self._settings.ollama_num_ctx,
                        },
                    },
                    headers={"X-Request-Id": request_id},
                    timeout=httpx.Timeout(
                        connect=min(5.0, restante),
                        read=restante,
                        write=min(10.0, restante),
                        pool=min(5.0, restante),
                    ),
                )
            except httpx.HTTPError as exc:
                ultimo_error = "No se pudo contactar a Ollama. ¿Está encendido?"
                logger.warning(
                    "ollama_network_error",
                    extra={
                        "event": "upstream_error",
                        "model": modelo,
                        "attempt": intento,
                        "error": str(exc)[:200],
                    },
                )
                if intento < self._settings.ollama_max_attempts:
                    await self._esperar(intento)
                    continue
                break

            if respuesta.status_code == 200:
                resultado = self._parsear_respuesta(
                    respuesta, attempt=intento, started=started
                )
                if resultado is not None:
                    return resultado
                ultimo_error = "La IA local no devolvió contenido."
                break

            detalle = _mensaje_error(respuesta)
            logger.warning(
                "ollama_upstream_error",
                extra={
                    "event": "upstream_error",
                    "model": modelo,
                    "attempt": intento,
                    "status": respuesta.status_code,
                    "upstream": detalle,
                },
            )

            if respuesta.status_code == 404:
                respaldo = self._settings.ollama_modelo_respaldo.strip()
                if respaldo and modelo != respaldo and intento < self._settings.ollama_max_attempts:
                    logger.warning(
                        "ollama_modelo_respaldo",
                        extra={
                            "event": "model_fallback",
                            "model": modelo,
                            "fallback": respaldo,
                        },
                    )
                    modelo = respaldo
                    continue
                raise OllamaError(
                    "El modelo local no existe. Crea 'tuki-chef' con el Modelfile "
                    "o descarga el modelo de respaldo.",
                    status_code=503,
                    code="ai_model_missing",
                )
            if respuesta.status_code == 429 or respuesta.status_code >= 500:
                ultimo_error = (
                    "La IA local está ocupada. Espera unos segundos."
                    if respuesta.status_code == 429
                    else "El servidor local de IA falló temporalmente."
                )
                if intento < self._settings.ollama_max_attempts:
                    await self._esperar(intento)
                    continue
                break

            ultimo_error = "La solicitud a la IA local no es válida."
            break

        logger.error(
            "ollama_exhausted",
            extra={
                "event": "upstream_exhausted",
                "model": self._settings.modelo_chat,
                "detail": ultimo_error,
            },
        )
        if agoto_tiempo:
            raise OllamaError(ultimo_error, status_code=504, code="ai_timeout")
        raise OllamaError(
            ultimo_error,
            status_code=429 if "ocupada" in ultimo_error else 502,
            code="ai_rate_limited" if "ocupada" in ultimo_error else "ai_unavailable",
            retry_after=5 if "ocupada" in ultimo_error else None,
        )

    async def embed(self, textos: list[str]) -> list[list[float]]:
        """Devuelve un vector por texto. Si Ollama falla, el índice sigue con BM25."""
        if not textos:
            return []
        if not self._settings.modelo_embeddings:
            return []

        vectores: list[list[float]] = []
        for inicio in range(0, len(textos), _LOTE_EMBEDDINGS):
            lote = textos[inicio : inicio + _LOTE_EMBEDDINGS]
            try:
                respuesta = await self._client.post(
                    self._url_embed,
                    json={
                        "model": self._settings.modelo_embeddings,
                        "input": lote,
                        "truncate": True,
                        "keep_alive": self._settings.ollama_keep_alive,
                    },
                    timeout=httpx.Timeout(
                        connect=5.0,
                        read=self._settings.ollama_timeout_seconds,
                        write=30.0,
                        pool=5.0,
                    ),
                )
            except httpx.HTTPError as exc:
                raise OllamaError(
                    "No se pudieron calcular los embeddings locales.",
                    status_code=503,
                    code="embeddings_unavailable",
                ) from exc

            if respuesta.status_code != 200:
                raise OllamaError(
                    "Ollama no pudo calcular los embeddings. "
                    f"¿Está descargado {self._settings.modelo_embeddings}?",
                    status_code=503,
                    code="embeddings_unavailable",
                )

            datos = respuesta.json()
            lote_vectores = datos.get("embeddings") if isinstance(datos, dict) else None
            if not isinstance(lote_vectores, list) or len(lote_vectores) != len(lote):
                raise OllamaError(
                    "Ollama devolvió embeddings con una forma inesperada.",
                    status_code=502,
                    code="embeddings_invalid",
                )
            vectores.extend([[float(valor) for valor in vector] for vector in lote_vectores])
        return vectores

    async def _esperar(self, intento: int) -> None:
        base = self._settings.ollama_backoff_seconds * (2 ** (intento - 1))
        retraso = base * (0.7 + random.random() * 0.6)
        retraso = min(retraso, self._settings.ollama_max_backoff_seconds)
        if retraso > 0:
            await asyncio.sleep(retraso)

    def _parsear_respuesta(
        self, respuesta: httpx.Response, *, attempt: int, started: float
    ) -> ResultadoIA | None:
        try:
            datos = respuesta.json()
        except ValueError:
            return None

        mensaje = datos.get("message") if isinstance(datos, dict) else None
        contenido = limpiar_salida(str((mensaje or {}).get("content") or ""))
        if not contenido:
            return None

        prompt_tokens = int(datos.get("prompt_eval_count") or 0)
        completion_tokens = int(datos.get("eval_count") or 0)
        latencia = time.perf_counter() - started
        modelo = str(datos.get("model") or self._settings.modelo_chat)

        return ResultadoIA(
            content=contenido,
            model=modelo,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            attempts=attempt,
            latency_ms=int(latencia * 1000),
        )


def _mensaje_error(respuesta: httpx.Response) -> str:
    try:
        datos = respuesta.json()
    except ValueError:
        return ""
    if isinstance(datos, dict):
        return str(datos.get("error", ""))[:300]
    return ""
