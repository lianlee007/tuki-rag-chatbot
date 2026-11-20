"""Genera una versión Markdown legible del catálogo."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
datos = json.loads((BASE / "datos" / "catalogo.json").read_text(encoding="utf-8"))

lineas = [
    "# Catálogo de restaurantes y platos",
    "",
    "Este documento es una versión legible del catálogo exportado de PostgreSQL.",
    "El chatbot usa el archivo `datos/catalogo.json`; este Markdown sirve para revisar",
    "y citar la base de conocimiento.",
    "",
]

for ficha in datos:
    restaurante = ficha["restaurante"]
    lineas.append(f"## {restaurante['nombre']}")
    lineas.append("")
    lineas.append(f"- Ciudad: {restaurante.get('ciudad') or 'Sin dato'}")
    lineas.append(f"- Dirección: {restaurante.get('direccion') or 'Sin dato'}")
    lineas.append(
        f"- Horario: {restaurante.get('horarioApertura')} a "
        f"{restaurante.get('horarioCierre')}"
    )
    if restaurante.get("descripcion"):
        lineas.append(f"- Descripción: {restaurante['descripcion']}")
    resumen = ficha.get("resumen_resenas") or {}
    if resumen.get("cantidad"):
        lineas.append(
            f"- Valoración: {resumen.get('promedio')} / 5 "
            f"({resumen.get('cantidad')} reseñas)"
        )
    lineas.append("")

    for platillo in ficha.get("platillos") or []:
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
        linea = (
            f"- **{platillo['nombre']}** — "
            f"{float(platillo['precio']):.0f} COP"
        )
        if platillo.get("descripcion"):
            linea += f". {platillo['descripcion']}"
        if etiquetas:
            linea += f" ({', '.join(etiquetas)})"
        if platillo.get("alergenos"):
            linea += f" Alérgenos: {', '.join(platillo['alergenos'])}."
        lineas.append(linea)
    lineas.append("")

(BASE / "conocimiento" / "catalogo.md").write_text(
    "\n".join(lineas), encoding="utf-8"
)
print(f"Documento generado con {len(datos)} restaurantes")
