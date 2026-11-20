"""Evaluador del chatbot RAG de TukiTuki.

Ejecuta el Golden Dataset contra la API local y calcula latencia, éxito por
palabras clave, fuentes citadas y abstención. Opcionalmente pide a Ollama una
nota de fidelidad de 1 a 5 (estrategia del juez) usando el mismo modelo local.

Uso:
    python evaluacion/evaluate_rag.py
    python evaluacion/evaluate_rag.py --limite 10
    python evaluacion/evaluate_rag.py --juez
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

MARCADORES_ABSTENCION = (
    "no puedo",
    "no lo manejo",
    "solo puedo",
    "no encontre",
    "no hay",
    "no tengo",
    "no dispongo",
    "no esta en",
    "no aparece",
    "no cuento",
    "no conozco",
    "no figure",
)


def normalizar(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", texto or "")
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return " ".join(sin_acentos.lower().split())


def cargar_dataset(ruta: Path) -> list[dict[str, Any]]:
    casos: list[dict[str, Any]] = []
    with ruta.open(encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if linea:
                casos.append(json.loads(linea))
    return casos


def claves_coinciden(respuesta: str, claves: list[str], modo: str) -> bool:
    if not claves:
        return True
    texto = normalizar(respuesta)
    encontradas = [clave for clave in claves if normalizar(clave) in texto]
    return len(encontradas) == len(claves) if modo == "todas" else bool(encontradas)


def se_abstiene(respuesta: str, fuentes: list[dict[str, Any]]) -> bool:
    if not fuentes:
        return True
    texto = normalizar(respuesta)
    return any(marcador in texto for marcador in MARCADORES_ABSTENCION)


def evaluar_caso(
    cliente: httpx.Client,
    caso: dict[str, Any],
    url_api: str,
    timeout: float,
) -> dict[str, Any]:
    inicio = time.perf_counter()
    respuesta = cliente.post(
        f"{url_api.rstrip('/')}/api/v1/ai/recomendar",
        json={
            "mensaje": caso["pregunta"],
            "historial": caso.get("historial") or [],
            **({"ciudad": caso["ciudad"]} if caso.get("ciudad") else {}),
            **(
                {"restaurante_slug": caso["restaurante_slug"]}
                if caso.get("restaurante_slug")
                else {}
            ),
        },
        timeout=timeout,
    )
    latencia_ms = int((time.perf_counter() - inicio) * 1000)

    if respuesta.status_code != 200:
        return {
            "id": caso["id"],
            "categoria": caso.get("categoria", ""),
            "pregunta": caso["pregunta"],
            "exito": False,
            "latencia_ms": latencia_ms,
            "error": f"HTTP {respuesta.status_code}: {respuesta.text[:200]}",
            "respuesta": "",
            "fuentes": [],
        }

    datos = respuesta.json()
    texto = str(datos.get("respuesta") or "")
    fuentes = datos.get("fuentes") or []
    claves_ok = claves_coinciden(
        texto,
        caso.get("palabras_clave") or [],
        caso.get("modo_clave", "todas"),
    )
    fuentes_ok = len(fuentes) >= int(caso.get("fuentes_min") or 0)
    abstencion_ok = True
    if caso.get("debe_abstenerse"):
        abstencion_ok = se_abstiene(texto, fuentes)

    exito = claves_ok and fuentes_ok and abstencion_ok
    return {
        "id": caso["id"],
        "categoria": caso.get("categoria", ""),
        "pregunta": caso["pregunta"],
        "exito": exito,
        "claves_ok": claves_ok,
        "fuentes_ok": fuentes_ok,
        "abstencion_ok": abstencion_ok,
        "latencia_ms": latencia_ms,
        "recuperacion_ms": datos.get("recuperacion_ms"),
        "fuentes": fuentes,
        "respuesta": texto,
        "modelo": datos.get("modelo"),
        "error": None,
    }


def juzgar(
    cliente: httpx.Client,
    pregunta: str,
    respuesta: str,
    fuentes: list[dict[str, Any]],
    modelo: str,
) -> int | None:
    contexto = "\n".join(
        f"- {fuente.get('nombre')} ({fuente.get('restaurante')}), "
        f"precio {fuente.get('precio')}"
        for fuente in fuentes[:6]
    )
    prompt = (
        "Evalúa la respuesta de un recomendador de platos con una nota de 1 a 5.\n"
        "Criterios: fidelidad a las fuentes (no inventa), completitud y claridad.\n"
        "Devuelve solo el número, sin texto adicional.\n\n"
        f"Pregunta: {pregunta}\n"
        f"Fuentes recuperadas:\n{contexto or 'ninguna'}\n"
        f"Respuesta: {respuesta}\n"
        "Nota:"
    )
    try:
        datos = cliente.post(
            "/api/chat",
            json={
                "model": modelo,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8},
                "keep_alive": "5m",
            },
            timeout=90.0,
        ).json()
        contenido = str((datos.get("message") or {}).get("content") or "")
        for caracter in contenido:
            if caracter in "12345":
                return int(caracter)
    except (httpx.HTTPError, ValueError):
        return None
    return None


def resumen(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    exitos = [caso for caso in resultados if caso["exito"]]
    latencias = sorted(caso["latencia_ms"] for caso in resultados)
    notas = [caso["juez"] for caso in resultados if caso.get("juez") is not None]

    def percentil(valor: float) -> int:
        if not latencias:
            return 0
        indice = min(len(latencias) - 1, max(0, round((len(latencias) - 1) * valor)))
        return int(latencias[indice])

    total = len(resultados)
    return {
        "total": total,
        "exitos": len(exitos),
        "fallos": total - len(exitos),
        "tasa_exito": round(100.0 * len(exitos) / total, 2) if total else 0.0,
        "tasa_error": round(100.0 * (total - len(exitos)) / total, 2) if total else 0.0,
        "latencia_media_ms": int(statistics.mean(latencias)) if latencias else 0,
        "latencia_p50_ms": percentil(0.50),
        "latencia_p95_ms": percentil(0.95),
        "juez_promedio": round(statistics.mean(notas), 2) if notas else None,
        "juez_casos": len(notas),
    }


def escribir_salidas(
    directorio: Path,
    resultados: list[dict[str, Any]],
    metricas: dict[str, Any],
    modelo_juez: str | None,
) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    (directorio / "resultados.json").write_text(
        json.dumps(
            {
                "generado_en": datetime.now(UTC).isoformat(),
                "modelo_juez": modelo_juez,
                "metricas": metricas,
                "casos": resultados,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (directorio / "resultados.csv").open("w", newline="", encoding="utf-8") as archivo:
        campos = [
            "id",
            "categoria",
            "exito",
            "latencia_ms",
            "recuperacion_ms",
            "claves_ok",
            "fuentes_ok",
            "abstencion_ok",
            "juez",
            "pregunta",
            "respuesta",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(resultados)

    lineas = [
        "# Resultados de la evaluación RAG",
        "",
        f"Generado: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Casos | {metricas['total']} |",
        f"| Éxito | {metricas['tasa_exito']} % |",
        f"| Error | {metricas['tasa_error']} % |",
        f"| Latencia media | {metricas['latencia_media_ms']} ms |",
        f"| Latencia p50 | {metricas['latencia_p50_ms']} ms |",
        f"| Latencia p95 | {metricas['latencia_p95_ms']} ms |",
    ]
    if metricas.get("juez_promedio") is not None:
        lineas.append(f"| Juez local (1-5) | {metricas['juez_promedio']} |")
    lineas.extend(
        [
            "",
            "## Casos fallidos",
            "",
        ]
    )
    fallidos = [caso for caso in resultados if not caso["exito"]]
    if not fallidos:
        lineas.append("Ninguno.")
    else:
        for caso in fallidos:
            lineas.append(
                f"- **{caso['id']}** ({caso['categoria']}): {caso['pregunta']} "
                f"→ claves={caso.get('claves_ok')}, fuentes={caso.get('fuentes_ok')}, "
                f"abstención={caso.get('abstencion_ok')}"
            )
    (directorio / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta el Golden Dataset del RAG de TukiTuki.")
    parser.add_argument("--url", default=os.getenv("CHATBOT_URL", "http://localhost:8000"))
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "golden_dataset.jsonl"),
    )
    parser.add_argument(
        "--salida",
        default=str(Path(__file__).parent / "resultados"),
    )
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--umbral", type=float, default=70.0)
    parser.add_argument("--juez", action="store_true", help="Pide una nota 1-5 al modelo local.")
    parser.add_argument(
        "--ollama",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--modelo-juez",
        default=os.getenv("OLLAMA_MODELO_CHAT", "tuki-chef"),
    )
    args = parser.parse_args()

    casos = cargar_dataset(Path(args.dataset))
    if args.limite:
        casos = casos[: args.limite]

    resultados: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout, trust_env=False) as cliente:
        for posicion, caso in enumerate(casos, start=1):
            resultado = evaluar_caso(cliente, caso, args.url, args.timeout)
            if args.juez and resultado["exito"]:
                with httpx.Client(base_url=args.ollama, timeout=90.0, trust_env=False) as juez:
                    resultado["juez"] = juzgar(
                        juez,
                        caso["pregunta"],
                        resultado["respuesta"],
                        resultado["fuentes"],
                        args.modelo_juez,
                    )
            resultados.append(resultado)
            estado = "OK" if resultado["exito"] else "FALLO"
            print(
                f"[{posicion:02d}/{len(casos)}] {resultado['id']} {estado} "
                f"{resultado['latencia_ms']} ms"
            )

    metricas = resumen(resultados)
    escribir_salidas(
        Path(args.salida),
        resultados,
        metricas,
        args.modelo_juez if args.juez else None,
    )

    print("\nResumen")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))
    if metricas["tasa_exito"] < args.umbral:
        print(f"\nAtención: {metricas['tasa_exito']} % < {args.umbral} % exigido.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
