"""Modelos simples del dominio del chatbot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResultadoIA:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: int
    latency_ms: int
