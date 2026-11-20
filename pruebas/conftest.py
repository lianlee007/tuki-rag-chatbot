from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "catalogo.json"


def build_settings(**overrides) -> Settings:
    base = {
        "environment": "test",
        "log_level": "WARNING",
        "ollama_url": "http://ollama.test",
        "ollama_modelo_chat": "tuki-chef",
        "ollama_modelo_respaldo": "qwen2.5:1.5b-instruct",
        "ollama_modelo_embeddings": "nomic-embed-text",
        "ollama_max_attempts": 2,
        "ollama_backoff_seconds": 0.0,
        "ollama_max_backoff_seconds": 0.0,
        "catalogo_path": str(FIXTURE),
        "conocimiento_ttl_seconds": 300,
    }
    base.update(overrides)
    return Settings(**base)


@asynccontextmanager
async def make_client(settings: Settings | None = None):
    app = create_app(settings or build_settings())
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://chatbot.test",
        ) as client:
            yield client


@pytest.fixture
def settings_factory():
    return build_settings


@pytest.fixture
def client_factory():
    return make_client
