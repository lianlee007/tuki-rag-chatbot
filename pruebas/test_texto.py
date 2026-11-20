from __future__ import annotations

from app.dominio.texto import chunk_texto, normalizar, tokenizar


def test_normaliza_acentos_y_plurales():
    assert normalizar("Pizzería Popayán") == "pizzeria popayan"
    assert "pizza" in tokenizar("Pizzas artesanales")


def test_chunk_largo_con_solapamiento():
    texto = " ".join(f"palabra{i}" for i in range(400))
    fragmentos = chunk_texto(texto, max_tokens=100, overlap=20)
    assert len(fragmentos) > 1
    assert fragmentos[0].split()[-1] in fragmentos[1].split()


def test_chunk_corto_no_se_parte():
    assert chunk_texto("Una arepa con queso.", max_tokens=220, overlap=40) == [
        "Una arepa con queso."
    ]
