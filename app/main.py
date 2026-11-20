"""API del chatbot RAG de TukiTuki.

Expone el chat, el estado del catálogo y una página web mínima. Está pensada
para correr sola con Ollama: no necesita el resto de microservicios.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.adaptadores.catalogo import ExtractorCatalogo
from app.adaptadores.ollama import OllamaClient
from app.config import Settings, get_settings
from app.dominio.errores import ErrorIA
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConocimientoResponse,
    FuenteResponse,
    HealthResponse,
    RecomendarRequest,
    RecomendarResponse,
    Usage,
)
from app.servicios.chat import ServicioChat
from app.servicios.conocimiento import ServicioConocimiento
from app.servicios.recomendador import ServicioRecomendador

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

SISTEMA_POR_DEFECTO = (
    "Eres un asistente gastronómico de TukiTuki. Respondes en español, "
    "de forma breve y concreta."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_or_raise()

    app = FastAPI(
        title="Tuki RAG Chatbot",
        version=settings.version,
        description="Chatbot local con RAG sobre el catálogo de TukiTuki.",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        application.state.settings = settings
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.ollama_timeout_seconds,
                write=30.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"User-Agent": f"{settings.app_name}/{settings.version}"},
            trust_env=False,
        )
        proveedor = OllamaClient(settings, client)
        conocimiento = ServicioConocimiento(
            ExtractorCatalogo(settings),
            proveedor,
            settings,
        )
        application.state.http = client
        application.state.conocimiento = conocimiento
        application.state.chat = ServicioChat(
            proveedor,
            max_chars=settings.request_max_chars,
        )
        application.state.recomendador = ServicioRecomendador(
            conocimiento,
            proveedor,
            proveedor,
            settings,
        )
        logger.info(
            "chatbot_started",
            extra={
                "event": "startup",
                "modelo_chat": settings.modelo_chat,
                "catalogo": settings.catalogo_path,
            },
        )
        try:
            yield
        finally:
            await client.aclose()

    app.router.lifespan_context = lifespan

    @app.exception_handler(ErrorIA)
    async def error_ia(_: Request, exc: ErrorIA) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=settings.version,
            environment=settings.environment,
        )

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        estado = request.app.state.conocimiento.estado()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "modelo_chat": settings.modelo_chat,
                "modelo_embeddings": settings.modelo_embeddings,
                "documentos": estado["documentos"],
                "embeddings": estado["embeddings"],
            },
        )

    @app.post("/api/v1/ai/recomendar", response_model=RecomendarResponse)
    async def recomendar(payload: RecomendarRequest, request: Request) -> RecomendarResponse:
        servicio: ServicioRecomendador = request.app.state.recomendador
        resultado = await servicio.recomendar(
            mensaje=payload.mensaje,
            historial=[turno.model_dump() for turno in payload.historial],
            ciudad=payload.ciudad,
            restaurante_slug=payload.restaurante_slug,
            request_id="local",
        )
        return RecomendarResponse(
            respuesta=resultado.content,
            fuentes=[
                FuenteResponse(
                    id=fuente.id,
                    tipo=fuente.tipo,
                    nombre=fuente.nombre,
                    restaurante=fuente.restaurante,
                    restaurante_slug=fuente.restaurante_slug,
                    ciudad=fuente.ciudad,
                    categoria=fuente.categoria,
                    precio=fuente.precio,
                    moneda=fuente.moneda,
                    imagen_url=fuente.imagen_url,
                    vegetariano=fuente.vegetariano,
                    vegano=fuente.vegano,
                    picante=fuente.picante,
                    alergenos=list(fuente.alergenos),
                )
                for fuente in resultado.fuentes
            ],
            modelo=resultado.model,
            usage=Usage(
                prompt_tokens=resultado.prompt_tokens,
                completion_tokens=resultado.completion_tokens,
                total_tokens=resultado.total_tokens,
            ),
            attempts=resultado.attempts,
            latency_ms=resultado.latency_ms,
            recuperacion_ms=resultado.recuperacion_ms,
            contexto_tokens=resultado.contexto_tokens,
            request_id="local",
        )

    @app.get("/api/v1/ai/conocimiento", response_model=ConocimientoResponse)
    async def conocimiento(request: Request) -> ConocimientoResponse:
        return ConocimientoResponse(**request.app.state.conocimiento.estado())

    @app.post("/api/v1/ai/conocimiento/recargar", response_model=ConocimientoResponse)
    async def recargar_conocimiento(request: Request) -> ConocimientoResponse:
        estado = await request.app.state.conocimiento.recargar()
        return ConocimientoResponse(**estado)

    @app.post("/api/v1/ai/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        servicio: ServicioChat = request.app.state.chat
        resultado = await servicio.generar(
            prompt=payload.prompt,
            system=payload.system or SISTEMA_POR_DEFECTO,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            request_id="local",
        )
        return ChatResponse(
            content=resultado.content,
            model=resultado.model,
            usage=Usage(
                prompt_tokens=resultado.prompt_tokens,
                completion_tokens=resultado.completion_tokens,
                total_tokens=resultado.total_tokens,
            ),
            attempts=resultado.attempts,
            latency_ms=resultado.latency_ms,
            request_id="local",
        )

    @app.get("/api/v1/ai/models")
    async def models() -> dict:
        return {
            "models": [settings.modelo_chat],
            "default": settings.modelo_chat,
            "embeddings": settings.modelo_embeddings,
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
