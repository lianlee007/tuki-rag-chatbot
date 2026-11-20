"""Recuperación híbrida para el recomendador.

La técnica es deliberadamente sencilla y barata: BM25 para las coincidencias
exactas (nombres, ingredientes, precios) y embeddings para las preguntas
semánticas. Los dos rankings se fusionan con Reciprocal Rank Fusion y luego se
elige un conjunto diverso con Maximal Marginal Relevance. Todo corre en CPU y
sin dependencias nuevas.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.dominio.conocimiento import Documento, Recuperado
from app.dominio.texto import (
    estimar_tokens,
    expandir_consulta,
    normalizar,
    tokenizar,
)

K_RRF = 60
PALABRAS_BARATO = ("barat", "economic", "asequible", "presupuesto")
TERMINOS_SIN_DATOS = (
    "china", "chino", "japonesa", "japones", "sushi", "mexicana", "mexicano",
    "tacos", "infantil", "ninos", "parqueadero",
)


@dataclass(frozen=True, slots=True)
class Filtros:
    vegetariano: bool | None = None
    vegano: bool | None = None
    picante: bool | None = None
    precio_min: float | None = None
    precio_max: float | None = None
    ciudad: str | None = None
    restaurante_slug: str | None = None
    excluir_alergeno: str | None = None
    categorias: tuple[str, ...] = ()


def _numero_cop(valor: str) -> float | None:
    limpio = re.sub(r"[^\d]", "", valor)
    if not limpio:
        return None
    numero = float(limpio)
    # "20" o "25" en una pregunta de precios suele significar miles.
    if numero < 1000:
        numero *= 1000
    return numero


def extraer_filtros(texto: str) -> Filtros:
    normalizado = normalizar(texto)
    vegetariano = None
    vegano = None
    picante = None

    if "vegan" in normalizado or "sin productos animales" in normalizado:
        vegano = True
        vegetariano = True
    elif "vegetarian" in normalizado or "veggie" in normalizado or "sin carne" in normalizado:
        vegetariano = True

    if any(palabra in normalizado for palabra in ("picante", "picoso", "aji", "chile")):
        picante = True

    precio_max: float | None = None
    precio_min: float | None = None
    patron_tope = re.search(
        r"(?:menos de|maximo|hasta|por debajo de|no mas de)\s*\$?\s*([\d\.,]+)",
        normalizado,
    )
    if patron_tope:
        precio_max = _numero_cop(patron_tope.group(1))
    patron_piso = re.search(
        r"(?:mas de|desde|minimo)\s*\$?\s*([\d\.,]+)",
        normalizado,
    )
    if patron_piso:
        precio_min = _numero_cop(patron_piso.group(1))
    if precio_max is None and any(palabra in normalizado for palabra in PALABRAS_BARATO):
        precio_max = 25000.0

    alergeno = None
    for candidato, equivalente in (
        ("gluten", "gluten"),
        ("lactosa", "lactosa"),
        ("lacteos", "lactosa"),
        ("mani", "mani"),
        ("nueces", "nueces"),
        ("mariscos", "mariscos"),
        ("huevo", "huevo"),
        ("soya", "soya"),
    ):
        if f"sin {candidato}" in normalizado:
            alergeno = equivalente
            break

    categorias: tuple[str, ...] = ()
    if any(palabra in normalizado for palabra in ("postre", "dulce", "torta", "helado")):
        categorias = ("Postres", "Repostería")
    elif any(
        palabra in normalizado
        for palabra in ("tomar", "bebida", "jugo", "limonada", "cafe")
    ):
        categorias = ("Café", "Bebidas", "Jugos", "Postres")

    return Filtros(
        vegetariano=vegetariano,
        vegano=vegano,
        picante=picante,
        precio_min=precio_min,
        precio_max=precio_max,
        excluir_alergeno=alergeno,
        categorias=categorias,
    )


class BM25:
    """BM25 Okapi sobre listas de tokens ya normalizados."""

    def __init__(self, corpus: Iterable[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._tokens = [list(tokens) for tokens in corpus]
        self._k1 = k1
        self._b = b
        self._tf = [Counter(tokens) for tokens in self._tokens]
        self._largos = [len(tokens) for tokens in self._tokens]
        self._promedio = (sum(self._largos) / len(self._largos)) if self._largos else 1.0
        self._df: Counter[str] = Counter()
        for tokens in self._tokens:
            self._df.update(set(tokens))
        self._total = max(1, len(self._tokens))

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        return math.log(1.0 + (self._total - df + 0.5) / (df + 0.5))

    def puntuar(self, consulta: list[tuple[str, float]]) -> list[float]:
        puntajes = [0.0] * len(self._tokens)
        for token, peso in consulta:
            idf = self._idf(token)
            if idf <= 0:
                continue
            for indice, tf in enumerate(self._tf):
                frecuencia = tf.get(token, 0)
                if not frecuencia:
                    continue
                largo = self._largos[indice] or 1
                denominador = frecuencia + self._k1 * (
                    1.0 - self._b + self._b * largo / (self._promedio or 1.0)
                )
                puntajes[indice] += peso * idf * (
                    frecuencia * (self._k1 + 1.0)
                ) / (denominador or 1.0)
        return puntajes


def coseno(a: Iterable[float], b: Iterable[float]) -> float:
    va = list(a)
    vb = list(b)
    if not va or not vb or len(va) != len(vb):
        return 0.0
    producto = sum(x * y for x, y in zip(va, vb, strict=False))
    norma_a = math.sqrt(sum(x * x for x in va))
    norma_b = math.sqrt(sum(y * y for y in vb))
    if norma_a <= 0 or norma_b <= 0:
        return 0.0
    return producto / (norma_a * norma_b)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    conjunto_a = set(a)
    conjunto_b = set(b)
    if not conjunto_a or not conjunto_b:
        return 0.0
    return len(conjunto_a & conjunto_b) / len(conjunto_a | conjunto_b)


def _claves_filtro(documento: Documento) -> dict[str, Any]:
    return documento.metadata


def _cumple_filtros(documento: Documento, filtros: Filtros) -> bool:
    datos = _claves_filtro(documento)
    if filtros.vegetariano is not None and bool(datos.get("vegetariano")) != filtros.vegetariano:
        return False
    if filtros.vegano is not None and bool(datos.get("vegano")) != filtros.vegano:
        return False
    if filtros.picante is not None and bool(datos.get("picante")) != filtros.picante:
        return False
    precio = float(datos.get("precio") or 0)
    if (
        filtros.precio_min is not None
        and documento.tipo == "plato"
        and precio < filtros.precio_min
    ):
        return False
    if (
        filtros.precio_max is not None
        and documento.tipo == "plato"
        and precio > filtros.precio_max
    ):
        return False
    if (
        filtros.restaurante_slug
        and str(datos.get("restaurante_slug") or "") != filtros.restaurante_slug
    ):
        return False
    if (
        filtros.ciudad
        and normalizar(filtros.ciudad) not in normalizar(str(datos.get("ciudad") or ""))
    ):
        return False
    if filtros.excluir_alergeno:
        alergenos = " ".join(str(item) for item in datos.get("alergenos") or [])
        if filtros.excluir_alergeno in normalizar(alergenos):
            return False
    if filtros.categorias:
        categoria = normalizar(str(datos.get("categoria") or ""))
        if not any(normalizar(nombre) in categoria for nombre in filtros.categorias):
            return False
    return True


class IndiceRAG:
    """Índice en memoria que combina BM25, vectores y metadatos."""

    def __init__(
        self,
        documentos: list[Documento],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.documentos = list(documentos)
        self.embeddings = list(embeddings or [])
        if self.embeddings and len(self.embeddings) != len(self.documentos):
            self.embeddings = []
        self._tokens = [tokenizar(documento.texto) for documento in self.documentos]
        self._bm25 = BM25(self._tokens)
        self._ciudades = sorted(
            {
                str(documento.metadata.get("ciudad"))
                for documento in self.documentos
                if documento.metadata.get("ciudad")
            }
        )
        self._restaurantes = sorted(
            {
                (
                    str(documento.metadata.get("restaurante")),
                    str(documento.metadata.get("restaurante_slug")),
                )
                for documento in self.documentos
                if documento.metadata.get("restaurante")
                and documento.metadata.get("restaurante_slug")
            }
        )
        self._platos = sorted(
            {
                (
                    str(documento.metadata.get("nombre")),
                    str(documento.metadata.get("id")),
                )
                for documento in self.documentos
                if documento.tipo == "plato"
                and documento.metadata.get("nombre")
                and documento.metadata.get("id")
            }
        )

    @property
    def tiene_embeddings(self) -> bool:
        return bool(self.embeddings)

    def _ciudad_detectada(self, consulta: str) -> str | None:
        normalizada = normalizar(consulta)
        for ciudad in self._ciudades:
            if normalizar(ciudad) and normalizar(ciudad) in normalizada:
                return ciudad
        return None

    def restaurante_detectado(self, consulta: str) -> str | None:
        """Devuelve el slug si la pregunta nombra un restaurante del catálogo."""
        normalizada = normalizar(consulta)
        for nombre, slug in self._restaurantes:
            nombre_normalizado = normalizar(nombre)
            if len(nombre_normalizado) >= 4 and nombre_normalizado in normalizada:
                return slug
        return None

    def documento_restaurante(self, slug: str) -> Documento | None:
        for documento in self.documentos:
            if (
                documento.tipo == "restaurante"
                and documento.metadata.get("restaurante_slug") == slug
            ):
                return documento
        return None

    def plato_detectado(self, consulta: str) -> str | None:
        """Devuelve el id del plato si la pregunta lo nombra con claridad."""
        normalizada = normalizar(consulta)
        for nombre, identificador in self._platos:
            nombre_normalizado = normalizar(nombre)
            if len(nombre_normalizado) >= 4 and nombre_normalizado in normalizada:
                return identificador
        return None

    def documento_plato(self, identificador: str) -> Documento | None:
        for documento in self.documentos:
            if documento.tipo == "plato" and documento.metadata.get("id") == identificador:
                return documento
        return None

    def buscar(
        self,
        consulta: str,
        *,
        vector_consulta: list[float] | None = None,
        filtros: Filtros | None = None,
        top_k: int = 6,
        candidatos: int = 20,
        mmr_lambda: float = 0.72,
        umbral_semantico: float = 0.22,
    ) -> list[Recuperado]:
        if not self.documentos:
            return []

        filtros = filtros or Filtros()
        ciudad = filtros.ciudad or self._ciudad_detectada(consulta)
        restaurante = filtros.restaurante_slug or self.restaurante_detectado(consulta)
        if (ciudad and not filtros.ciudad) or (
            restaurante and not filtros.restaurante_slug
        ):
            filtros = Filtros(
                vegetariano=filtros.vegetariano,
                vegano=filtros.vegano,
                picante=filtros.picante,
                precio_min=filtros.precio_min,
                precio_max=filtros.precio_max,
                ciudad=ciudad,
                restaurante_slug=restaurante,
                excluir_alergeno=filtros.excluir_alergeno,
                categorias=filtros.categorias,
            )

        elegibles = [
            indice
            for indice, documento in enumerate(self.documentos)
            if _cumple_filtros(documento, filtros)
        ]
        if not elegibles:
            return []

        expandidos = expandir_consulta(tokenizar(consulta))
        puntajes_lexicos = (
            self._bm25.puntuar(expandidos)
            if expandidos
            else [0.0] * len(self.documentos)
        )

        ranking_lexico = sorted(
            (indice for indice in elegibles if puntajes_lexicos[indice] > 0),
            key=lambda indice: puntajes_lexicos[indice],
            reverse=True,
        )[:candidatos]

        ranking_vector: list[int] = []
        similitudes: dict[int, float] = {}
        if vector_consulta and self.embeddings:
            for indice in elegibles:
                similitud = coseno(vector_consulta, self.embeddings[indice])
                similitudes[indice] = similitud
            ranking_vector = sorted(
                (
                    indice
                    for indice in elegibles
                    if similitudes.get(indice, 0.0) >= umbral_semantico
                ),
                key=lambda indice: similitudes[indice],
                reverse=True,
            )[:candidatos]

        # Sin ninguna señal, pero con una pregunta genérica de comida: se
        # ofrecen los destacados para que el modelo tenga algo real que citar.
        if not ranking_lexico and not ranking_vector:
            destacados = [
                indice
                for indice in elegibles
                if self.documentos[indice].tipo == "plato"
                and self.documentos[indice].metadata.get("destacado")
            ]
            platos = [i for i in elegibles if self.documentos[i].tipo == "plato"]
            ranking_lexico = (destacados or platos)[:candidatos]

        fusion: dict[int, float] = {}
        for ranking in (ranking_lexico, ranking_vector):
            for posicion, indice in enumerate(ranking, start=1):
                fusion[indice] = fusion.get(indice, 0.0) + 1.0 / (K_RRF + posicion)

        if not fusion:
            return []

        ordenados = sorted(fusion, key=lambda indice: fusion[indice], reverse=True)
        max_fusion = max(fusion.values()) or 1.0
        relevancia = {indice: fusion[indice] / max_fusion for indice in ordenados}

        # Diversidad: reparte los cupos entre los restaurantes disponibles.
        seleccionados: list[int] = []
        por_restaurante: Counter[str] = Counter()
        total_restaurantes = max(
            1,
            len(
                {
                    self.documentos[indice].metadata.get("restaurante_slug")
                    for indice in elegibles
                    if self.documentos[indice].metadata.get("restaurante_slug")
                }
            ),
        )
        max_por_restaurante = max(1, math.ceil(top_k / total_restaurantes))

        while ordenados and len(seleccionados) < top_k:
            mejor_indice = None
            mejor_puntaje = -1.0
            for indice in ordenados:
                documento = self.documentos[indice]
                restaurante = str(documento.metadata.get("restaurante_slug") or "")
                if por_restaurante[restaurante] >= max_por_restaurante:
                    continue
                similitud_max = 0.0
                for elegido in seleccionados:
                    if self.embeddings:
                        similitud = coseno(self.embeddings[indice], self.embeddings[elegido])
                    else:
                        similitud = jaccard(self._tokens[indice], self._tokens[elegido])
                    similitud_max = max(similitud_max, similitud)
                mmr = mmr_lambda * relevancia[indice] - (1.0 - mmr_lambda) * similitud_max
                if mmr > mejor_puntaje:
                    mejor_puntaje = mmr
                    mejor_indice = indice
            if mejor_indice is None:
                break
            ordenados.remove(mejor_indice)
            seleccionados.append(mejor_indice)
            restaurante = str(self.documentos[mejor_indice].metadata.get("restaurante_slug") or "")
            por_restaurante[restaurante] += 1

        return [
            Recuperado(
                documento=self.documentos[indice],
                puntaje=round(fusion[indice], 6),
                origen="lexico+vector" if ranking_vector else "lexico",
            )
            for indice in seleccionados[:top_k]
        ]

    def termino_sin_datos(self, consulta: str) -> str | None:
        """Detecta una cocina o servicio que el catálogo no tiene."""
        normalizada = normalizar(consulta).split()
        presentes = {
            palabra
            for documento in self.documentos
            if documento.tipo == "plato"
            for palabra in normalizar(documento.texto).split()
        }
        for termino in TERMINOS_SIN_DATOS:
            if termino in normalizada and termino not in presentes:
                return termino
        return None

    def estadisticas(self) -> dict[str, Any]:
        return {
            "documentos": len(self.documentos),
            "platos": sum(1 for documento in self.documentos if documento.tipo == "plato"),
            "restaurantes": len(
                {
                    documento.metadata.get("restaurante_id")
                    for documento in self.documentos
                    if documento.metadata.get("restaurante_id")
                }
            ),
            "embeddings": len(self.embeddings),
            "ciudades": self._ciudades,
        }


def seleccionar_contexto(
    recuperados: list[Recuperado], max_tokens: int
) -> tuple[str, list[Recuperado]]:
    """Formatea el contexto y recorta si el modelo se quedaría sin ventana."""
    bloques: list[str] = []
    usados: list[Recuperado] = []
    total = 0
    for posicion, recuperado in enumerate(recuperados, start=1):
        metadata = recuperado.documento.metadata
        etiqueta = metadata.get("nombre") or recuperado.documento.id
        bloque = f"[P{posicion}] {etiqueta}. {recuperado.documento.texto}"
        tokens = estimar_tokens(bloque)
        if usados and total + tokens > max_tokens:
            break
        bloques.append(bloque)
        usados.append(recuperado)
        total += tokens
    return "\n\n".join(bloques), usados


def similitud_texto(a: str, b: str) -> float:
    return jaccard(tokenizar(a), tokenizar(b))


def formatear_pesos(consulta: str) -> list[tuple[str, float]]:
    """Expuesto para pruebas y para depurar por qué se recuperó un plato."""
    return expandir_consulta(tokenizar(consulta))


__all__ = [
    "BM25",
    "Filtros",
    "IndiceRAG",
    "coseno",
    "extraer_filtros",
    "formatear_pesos",
    "jaccard",
    "seleccionar_contexto",
    "similitud_texto",
]
