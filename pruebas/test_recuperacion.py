from __future__ import annotations

from app.dominio.conocimiento import Documento, Recuperado
from app.dominio.recuperacion import (
    BM25,
    Filtros,
    IndiceRAG,
    coseno,
    extraer_filtros,
    seleccionar_contexto,
)
from app.dominio.texto import tokenizar


def _documentos() -> list[Documento]:
    return [
        Documento(
            "plato:1",
            "plato",
            "Plato: Arepa de queso. Restaurante: Café Bolívar. Precio: 12000 COP.",
            {
                "id": "1",
                "nombre": "Arepa de queso",
                "restaurante_slug": "cafe-bolivar",
                "restaurante": "Café Bolívar",
                "ciudad": "Popayán",
                "precio": 12000.0,
                "vegetariano": True,
                "vegano": False,
                "picante": False,
                "alergenos": ["lácteos"],
            },
        ),
        Documento(
            "plato:2",
            "plato",
            "Plato: Pizza de pepperoni. Restaurante: Pizzería Centenario. Precio: 32000 COP.",
            {
                "id": "2",
                "nombre": "Pizza de pepperoni",
                "restaurante_slug": "pizzeria-centenario",
                "restaurante": "Pizzería Centenario",
                "ciudad": "Popayán",
                "precio": 32000.0,
                "vegetariano": False,
                "vegano": False,
                "picante": False,
                "alergenos": ["gluten", "lácteos"],
            },
        ),
        Documento(
            "plato:3",
            "plato",
            "Plato: Pizza vegetariana. Restaurante: Pizzería Centenario. Precio: 27000 COP.",
            {
                "id": "3",
                "nombre": "Pizza vegetariana",
                "restaurante_slug": "pizzeria-centenario",
                "restaurante": "Pizzería Centenario",
                "ciudad": "Popayán",
                "precio": 27000.0,
                "vegetariano": True,
                "vegano": False,
                "picante": False,
                "alergenos": ["gluten", "lácteos"],
            },
        ),
    ]


def test_bm25_prioriza_el_nombre_del_plato():
    bm25 = BM25([tokenizar(d.texto) for d in _documentos()])
    puntajes = bm25.puntuar([("pizza", 1.0)])
    assert puntajes[1] > puntajes[0]


def test_indice_filtra_vegetariano_y_precio():
    indice = IndiceRAG(_documentos())
    resultados = indice.buscar(
        "algo vegetariano barato",
        filtros=Filtros(vegetariano=True, precio_max=25000),
        top_k=3,
    )
    assert [r.documento.id for r in resultados] == ["plato:1"]


def test_indice_detecta_la_ciudad_y_excluye_alergenos():
    indice = IndiceRAG(_documentos())
    filtros = extraer_filtros("pizza en Popayán sin gluten")
    resultados = indice.buscar(
        "pizza en Popayán sin gluten", filtros=filtros, top_k=3
    )
    assert resultados
    assert all(
        "gluten" not in r.documento.metadata["alergenos"] for r in resultados
    )


def test_detecta_cocina_que_no_esta_en_los_platos():
    indice = IndiceRAG(_documentos())
    assert indice.termino_sin_datos("¿Hay comida china?") == "china"
    assert indice.termino_sin_datos("¿Tienen sushi?") == "sushi"
    assert indice.termino_sin_datos("Quiero pizza") is None


def test_extraer_filtros_de_lenguaje_natural():
    filtros = extraer_filtros("Quiero algo vegano, picante y por menos de 25.000")
    assert filtros.vegano is True
    assert filtros.vegetariano is True
    assert filtros.picante is True
    assert filtros.precio_max == 25000


def test_reconoce_bebidas_y_postres():
    assert extraer_filtros("Quiero algo para beber").categorias
    assert extraer_filtros("Algo dulce de postre").categorias


def test_coseno_y_contexto():
    assert coseno([1, 0], [1, 0]) == 1.0
    assert coseno([1, 0], [0, 1]) == 0.0
    contexto, usados = seleccionar_contexto(
        [
            Recuperado(_documentos()[0], 1.0, "lexico"),
            Recuperado(_documentos()[1], 0.5, "lexico"),
        ],
        max_tokens=20,
    )
    assert contexto.startswith("[P1]")
    assert len(usados) == 1
