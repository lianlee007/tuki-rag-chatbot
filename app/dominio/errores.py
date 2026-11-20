"""Errores que la API traduce a respuestas HTTP."""

from __future__ import annotations


class ErrorIA(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "ai_unavailable",
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after


class PromptDemasiadoLargo(ErrorIA):
    def __init__(self, max_chars: int) -> None:
        super().__init__(
            f"El texto supera el máximo de {max_chars} caracteres.",
            status_code=413,
            code="prompt_too_long",
        )
