from __future__ import annotations

import json

import httpx
import respx
from conftest import make_client

OLLAMA = "http://ollama.test"


def montar_ollama(chat_content: str = "Te recomiendo la Arepa de queso [P1]."):
    respx.post(f"{OLLAMA}/api/embed").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "embeddings": [
                    [1.0, float(indice % 3), 0.5]
                    for indice, _ in enumerate(json.loads(request.content)["input"])
                ]
            },
        )
    )
    return respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "tuki-chef",
                "message": {"content": chat_content},
                "prompt_eval_count": 180,
                "eval_count": 25,
            },
        )
    )


@respx.mock
async def test_filtro_vegetariano_responde_lista_sin_llm():
    chat = montar_ollama()
    async with make_client() as client:
        response = await client.post(
            "/api/v1/ai/recomendar",
            json={"mensaje": "Quiero algo vegetariano y barato"},
        )
        estado = await client.get("/api/v1/ai/conocimiento")

    assert response.status_code == 200
    body = response.json()
    assert "Arepa de queso" in body["respuesta"]
    assert "lácteos" in body["respuesta"]
    nombres = [fuente["nombre"] for fuente in body["fuentes"]]
    assert "Arepa de queso" in nombres
    arepa = next(f for f in body["fuentes"] if f["nombre"] == "Arepa de queso")
    assert arepa["alergenos"] == ["lácteos"]
    assert chat.calls == []
    assert estado.json()["documentos"] >= 3


@respx.mock
async def test_recomienda_con_modelo_local_y_fuentes():
    chat = montar_ollama("Te recomiendo la Arepa de queso [P1] del Café Bolívar.")
    async with make_client() as client:
        response = await client.post(
            "/api/v1/ai/recomendar",
            json={"mensaje": "¿Qué me recomiendas?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "Arepa de queso" in body["respuesta"]
    assert body["usage"]["prompt_tokens"] == 180
    enviado = json.loads(chat.calls[0].request.content)
    prompt = enviado["messages"][1]["content"]
    assert "CONTEXTO DEL CATÁLOGO" in prompt
    assert "Arepa de queso" in prompt


@respx.mock
async def test_pregunta_ajena_no_consume_el_modelo():
    chat = montar_ollama()
    async with make_client() as client:
        response = await client.post(
            "/api/v1/ai/recomendar",
            json={"mensaje": "¿Quién ganó el partido de fútbol?"},
        )
    assert response.status_code == 200
    assert response.json()["fuentes"] == []
    assert chat.calls == []


async def test_health_y_modelos():
    async with make_client() as client:
        health = await client.get("/healthz")
        ready = await client.get("/readyz")
        models = await client.get("/api/v1/ai/models")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert models.json()["default"] == "tuki-chef"
