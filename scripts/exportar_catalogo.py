"""Exporta el catálogo de restaurantes a un JSON reproducible.

Uso:
    python scripts/exportar_catalogo.py
    RESTAURANTES_URL=http://localhost:3002 python scripts/exportar_catalogo.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent.parent
DESTINO = BASE / "datos" / "catalogo.json"
RESTAURANTES = os.getenv("RESTAURANTES_URL", "http://localhost:3002").rstrip("/")
RESENAS = os.getenv("RESENAS_URL", "http://localhost:3005").rstrip("/")


def main() -> None:
    fichas: list[dict] = []
    with httpx.Client(timeout=30.0, trust_env=False) as cliente:
        respuesta = cliente.get(f"{RESTAURANTES}/v1/restaurantes", params={"limite": 60, "desde": 0})
        respuesta.raise_for_status()
        restaurantes = respuesta.json().get("items") or []

        for restaurante in restaurantes:
            slug = restaurante.get("slug")
            if not slug:
                continue
            ficha = cliente.get(f"{RESTAURANTES}/v1/restaurantes/{slug}")
            if ficha.status_code != 200:
                print(f"omitido: {slug} ({ficha.status_code})")
                continue
            datos = ficha.json()
            resumen = {"promedio": 0, "cantidad": 0}
            try:
                valoraciones = cliente.get(
                    f"{RESENAS}/v1/resenas/restaurante/{datos['restaurante']['id']}"
                )
                if valoraciones.status_code == 200:
                    resumen = valoraciones.json().get("resumen") or resumen
            except httpx.HTTPError:
                pass
            datos["resumen_resenas"] = resumen
            fichas.append(datos)
            print(f"agregado: {slug}")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(fichas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(fichas)} restaurantes escritos en {DESTINO}")


if __name__ == "__main__":
    main()
