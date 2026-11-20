from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
datos = json.loads((BASE / "evaluacion" / "resultados" / "resultados.json").read_text(encoding="utf-8"))
metricas = datos["metricas"]
casos = datos["casos"]
por_categoria = defaultdict(lambda: [0, 0])
for caso in casos:
    por_categoria[caso["categoria"]][1] += 1
    if caso["exito"]:
        por_categoria[caso["categoria"]][0] += 1

filas = "\n".join(
    f"<tr><td>{cat}</td><td>{ok}/{total}</td><td>{100 * ok / total:.1f} %</td></tr>"
    for cat, (ok, total) in sorted(por_categoria.items())
)
fallos = "\n".join(
    f"<tr><td>{c['id']}</td><td>{c['pregunta']}</td><td>{c['respuesta'][:180]}</td></tr>"
    for c in casos
    if not c["exito"]
) or '<tr><td colspan="3">Sin fallos.</td></tr>'

html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Evaluación RAG TukiTuki</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; background: #FDFBF7; color: #2B2421; margin: 0; padding: 40px; }}
  h1 {{ font-size: 30px; margin: 0 0 8px; }}
  .sub {{ color: #786E68; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; max-width: 980px; }}
  .card {{ background: #fff; border: 1px solid #E7DAD3; border-radius: 18px; padding: 18px; }}
  .card b {{ display: block; font-size: 30px; color: #B34A2E; }}
  .card span {{ color: #786E68; font-size: 13px; }}
  table {{ width: 100%; max-width: 980px; border-collapse: collapse; margin-top: 24px; background: #fff; border-radius: 14px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #E7DAD3; font-size: 13px; vertical-align: top; }}
  th {{ background: #F5EEE6; }}
  h2 {{ margin-top: 34px; font-size: 20px; }}
  .ok {{ color: #2F5D36; font-weight: 700; }}
</style></head><body>
<h1>Evaluación del chatbot RAG</h1>
<div class="sub">TukiTuki · modelo <b>tuki-chef</b> · embeddings <b>nomic-embed-text</b> · Golden Dataset de 52 casos</div>
<div class="grid">
  <div class="card"><b>{metricas['tasa_exito']} %</b><span>Éxito ({metricas['exitos']}/{metricas['total']})</span></div>
  <div class="card"><b>{metricas['tasa_error']} %</b><span>Error</span></div>
  <div class="card"><b>{metricas['latencia_media_ms']} ms</b><span>Latencia media</span></div>
  <div class="card"><b>{metricas['latencia_p50_ms']} ms</b><span>Latencia p50</span></div>
  <div class="card"><b>{metricas['latencia_p95_ms']} ms</b><span>Latencia p95</span></div>
  <div class="card"><b class="ok">≥ 70 %</b><span>Umbral de aceptación</span></div>
</div>
<h2>Resultados por categoría</h2>
<table><tr><th>Categoría</th><th>Éxitos</th><th>Tasa</th></tr>{filas}</table>
<h2>Casos fallidos</h2>
<table><tr><th>ID</th><th>Pregunta</th><th>Respuesta</th></tr>{fallos}</table>
</body></html>"""
(BASE / "docs" / "resultados-evaluacion.html").write_text(html, encoding="utf-8")
print("HTML de evaluación generado")
