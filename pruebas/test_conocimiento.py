from __future__ import annotations

from app.config import Settings
from app.dominio.conocimiento import Documento
from app.servicios.conocimiento import ServicioConocimiento


class ExtractorFijo:
    def __init__(self, documentos: list[Documento]) -> None:
        self.documentos = documentos

    async def extraer(self) -> list[Documento]:
        return list(self.documentos)


class EmbeddingsContador:
    def __init__(self) -> None:
        self.llamadas = 0

    async def embed(self, textos: list[str]) -> list[list[float]]:
        self.llamadas += 1
        return [[float(indice), 0.5] for indice, _ in enumerate(textos)]


def _settings() -> Settings:
    return Settings(environment="test", log_level="WARNING")


async def test_reutiliza_embeddings_si_el_catalogo_no_cambia():
    documentos = [
        Documento("plato:1", "plato", "Plato: Arepa de queso.", {"id": "1"}),
    ]
    extractor = ExtractorFijo(documentos)
    embeddings = EmbeddingsContador()
    servicio = ServicioConocimiento(extractor, embeddings, _settings())

    await servicio.obtener()
    servicio._actualizado = 0  # fuerza el refresco por TTL
    await servicio.obtener()
    assert embeddings.llamadas == 1

    extractor.documentos = [
        Documento("plato:2", "plato", "Plato: Pizza margarita.", {"id": "2"}),
    ]
    servicio._actualizado = 0
    await servicio.obtener()
    assert embeddings.llamadas == 2
