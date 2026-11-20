from pathlib import Path

p = Path("docs/informe.md")
t = p.read_text(encoding="utf-8")
reemplazos = {
    "**70 pruebas**": "**13 pruebas**",
    "Extracción del catálogo y caché en disco.": "Extracción del catálogo y lectura del JSON.",
    "Rate limit, autenticación y errores del proveedor.": "Errores del proveedor y respuestas de la API.",
    "**70/70 pruebas correctas**": "**13/13 pruebas correctas**",
    "`services/ai-gateway/evaluacion/golden_dataset.jsonl`": "`evaluacion/golden_dataset.jsonl`",
    "| Éxitos | 51 |": "| Éxitos | 52 |",
    "| Tasa de éxito | **98.08 %** |": "| Tasa de éxito | **100 %** |",
    "| Tasa de error | 1.92 % |": "| Tasa de error | 0 % |",
    "| Latencia media | 2.639 ms |": "| Latencia media | 1.701 ms |",
    "| Latencia p50 | 93 ms |": "| Latencia p50 | 62 ms |",
    "| Latencia p95 | 8.718 ms |": "| Latencia p95 | 9.113 ms |",
    "subió a 98.08 %": "subió a 100 %",
    "La categoría de políticas quedó en\n80 % por ese caso.": "Todas las categorías quedaron al 100 %.",
}
for viejo, nuevo in reemplazos.items():
    t = t.replace(viejo, nuevo)

parrafo_viejo = (
    "El único fallo fue el caso `g49`, “¿Me puedes decir el precio exacto de todo el\n"
    "menú?”, donde el bot pidió una ciudad en vez de listar todo el menú. Es un caso\n"
    "de alcance poco realista: el menú completo son 61 platos y el contexto está\n"
    "limitado a 6 fragmentos. Se deja como mejora futura."
)
parrafo_nuevo = (
    "En esta corrida los 52 casos pasaron. La mejora clave fue filtrar por categoría\n"
    "cuando la pregunta pide un postre o algo para tomar; antes el modelo recomendaba\n"
    "una pizza para beber."
)
t = t.replace(parrafo_viejo, parrafo_nuevo)
p.write_text(t, encoding="utf-8")
print("Informe actualizado")
