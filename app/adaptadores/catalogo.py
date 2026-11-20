"""Extrae los documentos del catálogo exportado de PostgreSQL.

En TukiTuki el gateway consulta `restaurantes-service` por HTTP. Para que este
repositorio sea autocontenido, aquí se lee `datos/catalogo.json`, que es una foto
real de esa base de datos.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.dominio.conocimiento import Documento
from app.dominio.errores import ErrorIA
from app.dominio.texto import chunk_texto

logger = logging.getLogger(__name__)


class ExtractorCatalogo:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def extraer(self) -> list[Documento]:
        ruta = Path(self._settings.catalogo_path)
        try:
            datos = await asyncio.to_thread(_leer_json, ruta)
        except FileNotFoundError as exc:
            raise ErrorIA(
                f"No se encontró el catálogo en {ruta}.",
                status_code=503,
                code="knowledge_unavailable",
            ) from exc
        except (OSError, ValueError) as exc:
            raise ErrorIA(
                "El catálogo no se pudo leer.",
                status_code=503,
                code="knowledge_unavailable",
            ) from exc

        if not isinstance(datos, list):
            raise ErrorIA(
                "El catálogo debe ser una lista de restaurantes.",
                status_code=503,
                code="knowledge_invalid",
            )

        documentos: list[Documento] = []
        for ficha in datos:
            if not isinstance(ficha, dict):
                continue
            documentos.extend(self._documentos_de_ficha(ficha))
        return documentos

    def _documentos_de_ficha(self, ficha: dict[str, Any]) -> list[Documento]:
        restaurante = ficha.get("restaurante") or {}
        if not isinstance(restaurante, dict):
            return []
        cocinas = [
            str(item.get("nombre"))
            for item in ficha.get("cocinas") or []
            if isinstance(item, dict) and item.get("nombre")
        ]
        categorias = {
            str(item.get("id")): str(item.get("nombre"))
            for item in ficha.get("categorias") or []
            if isinstance(item, dict) and item.get("id")
        }
        valoracion = ficha.get("resumen_resenas") or {}
        if not isinstance(valoracion, dict):
            valoracion = {}

        documentos = self._documentos_restaurante(restaurante, cocinas, valoracion)
        for platillo in ficha.get("platillos") or []:
            if isinstance(platillo, dict):
                documentos.extend(
                    self._documentos_platillo(platillo, restaurante, categorias)
                )
        return documentos

    def _documentos_restaurante(
        self,
        restaurante: dict[str, Any],
        cocinas: list[str],
        valoracion: dict[str, Any],
    ) -> list[Documento]:
        nombre = str(restaurante.get("nombre") or "Restaurante")
        ciudad = _texto(restaurante.get("ciudad"))
        direccion = _texto(restaurante.get("direccion"))
        horario = _horario(restaurante)
        descripcion = _texto(restaurante.get("descripcion"))
        eslogan = _texto(restaurante.get("eslogan"))

        partes = [
            f"Restaurante: {nombre}.",
            f"Ciudad: {ciudad}." if ciudad else "",
            f"Cocinas: {', '.join(cocinas)}." if cocinas else "",
            f"Eslogan: {eslogan}." if eslogan else "",
            f"Descripción: {descripcion}." if descripcion else "",
            f"Horario: {horario}." if horario else "",
            f"Dirección: {direccion}." if direccion else "",
            f"Teléfono: {_texto(restaurante.get('telefono'))}."
            if restaurante.get("telefono")
            else "",
        ]
        if valoracion.get("cantidad"):
            partes.append(
                f"Valoración: {float(valoracion.get('promedio') or 0):.1f} de 5 "
                f"con {int(valoracion.get('cantidad') or 0)} reseñas."
            )
        texto = " ".join(parte for parte in partes if parte)
        metadata = {
            "id": str(restaurante.get("id") or ""),
            "tipo": "restaurante",
            "nombre": nombre,
            "restaurante": nombre,
            "restaurante_id": str(restaurante.get("id") or ""),
            "restaurante_slug": str(restaurante.get("slug") or ""),
            "ciudad": ciudad,
            "direccion": direccion,
            "horario": horario,
            "telefono": _texto(restaurante.get("telefono")),
            "whatsapp": _texto(restaurante.get("whatsapp")),
            "cocinas": cocinas,
            "calificacion": valoracion.get("promedio"),
            "resenas_cantidad": valoracion.get("cantidad"),
            "imagen_url": restaurante.get("imagenBannerUrl")
            or restaurante.get("imagenLogoUrl"),
        }
        return [
            Documento(
                id=f"restaurante:{metadata['id']}" + (f"#{posicion}" if posicion else ""),
                tipo="restaurante",
                texto=fragmento,
                metadata=metadata,
            )
            for posicion, fragmento in enumerate(
                chunk_texto(
                    texto,
                    max_tokens=self._settings.rag_chunk_tokens,
                    overlap=self._settings.rag_chunk_overlap,
                )
            )
        ]

    def _documentos_platillo(
        self,
        platillo: dict[str, Any],
        restaurante: dict[str, Any],
        categorias: dict[str, str],
    ) -> list[Documento]:
        nombre = str(platillo.get("nombre") or "Plato")
        restaurante_nombre = str(restaurante.get("nombre") or "Restaurante")
        ciudad = _texto(restaurante.get("ciudad"))
        categoria = categorias.get(str(platillo.get("categoriaId") or ""), "")
        precio = _precio(platillo.get("precio"))
        descripcion = _texto(platillo.get("descripcion"))
        ingredientes = _lista(platillo.get("ingredientes"))
        alergenos = _lista(platillo.get("alergenos"))
        etiquetas = [
            etiqueta
            for etiqueta, activo in (
                ("vegetariano", platillo.get("vegetariano")),
                ("vegano", platillo.get("vegano")),
                ("picante", platillo.get("picante")),
                ("destacado", platillo.get("destacado")),
            )
            if activo
        ]
        tiempo = platillo.get("tiempoPreparacion")

        partes = [
            f"Plato: {nombre}.",
            f"Restaurante: {restaurante_nombre}.",
            f"Ciudad: {ciudad}." if ciudad else "",
            f"Categoría: {categoria}." if categoria else "",
            f"Precio: {precio:.0f} COP." if precio is not None else "",
            f"Descripción: {descripcion}." if descripcion else "",
            f"Ingredientes: {', '.join(ingredientes)}." if ingredientes else "",
            f"Alérgenos: {', '.join(alergenos)}."
            if alergenos
            else "Sin alérgenos declarados.",
            f"Etiquetas: {', '.join(etiquetas)}." if etiquetas else "",
            f"Tiempo de preparación: {tiempo} minutos." if tiempo else "",
        ]
        texto = " ".join(parte for parte in partes if parte)
        metadata = {
            "id": str(platillo.get("id") or ""),
            "tipo": "plato",
            "nombre": nombre,
            "restaurante": restaurante_nombre,
            "restaurante_id": str(restaurante.get("id") or ""),
            "restaurante_slug": str(restaurante.get("slug") or ""),
            "ciudad": ciudad,
            "categoria": categoria or None,
            "descripcion": descripcion,
            "precio": precio,
            "moneda": "COP",
            "imagen_url": platillo.get("imagenUrl"),
            "vegetariano": bool(platillo.get("vegetariano")),
            "vegano": bool(platillo.get("vegano")),
            "picante": bool(platillo.get("picante")),
            "destacado": bool(platillo.get("destacado")),
            "ingredientes": ingredientes,
            "alergenos": alergenos,
        }
        return [
            Documento(
                id=f"plato:{metadata['id']}" + (f"#{posicion}" if posicion else ""),
                tipo="plato",
                texto=fragmento,
                metadata=metadata,
            )
            for posicion, fragmento in enumerate(
                chunk_texto(
                    texto,
                    max_tokens=self._settings.rag_chunk_tokens,
                    overlap=self._settings.rag_chunk_overlap,
                )
            )
        ]


def _leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _texto(valor: Any) -> str:
    return str(valor).strip() if valor is not None else ""


def _lista(valor: Any) -> list[str]:
    if not isinstance(valor, list):
        return []
    return [str(item).strip() for item in valor if str(item).strip()]


def _precio(valor: Any) -> float | None:
    try:
        return float(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None


def _horario(restaurante: dict[str, Any]) -> str:
    apertura = _texto(restaurante.get("horarioApertura"))
    cierre = _texto(restaurante.get("horarioCierre"))
    if apertura and cierre:
        return f"de {apertura} a {cierre}"
    return ""
