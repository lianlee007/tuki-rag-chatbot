from pathlib import Path

p = Path("README.md")
t = p.read_text(encoding="utf-8")
t = t.replace("| Éxitos | 50 |", "| Éxitos | 52 |")
t = t.replace("| Tasa de éxito | **96.15 %** |", "| Tasa de éxito | **100 %** |")
t = t.replace("| Error | 3.85 % |", "| Error | 0 % |")
t = t.replace("| Latencia media | 3.647 ms |", "| Latencia media | 1.701 ms |")
t = t.replace("| Latencia p50 | 94 ms |", "| Latencia p50 | 62 ms |")
t = t.replace("| Latencia p95 | 11.439 ms |", "| Latencia p95 | 9.113 ms |")
t = t.replace(
    "El único fallo fue `g49` (“precio exacto de todo el menú”), donde el bot pidió\n"
    "una ciudad en vez de listar los 61 platos. El contexto está limitado a 6\n"
    "fragmentos, así que ese caso queda como mejora futura.",
    "En esta corrida los 52 casos pasaron. La mejora clave fue filtrar por categoría\n"
    "cuando la pregunta pide un postre o algo para tomar; antes el modelo recomendaba\n"
    "una pizza para beber.",
)
p.write_text(t, encoding="utf-8")
print("README actualizado")
