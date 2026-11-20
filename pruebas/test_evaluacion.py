from __future__ import annotations

from evaluacion.evaluate_rag import resumen


def test_resumen_mide_palabras_clave_fuentes_y_abstenciones_aplicables():
    resultados = [
        {
            "exito": True,
            "latencia_ms": 10,
            "palabras_clave_requeridas": 2,
            "claves_ok": True,
            "fuentes_requeridas": 1,
            "fuentes_ok": True,
            "debe_abstenerse": False,
            "abstencion_ok": True,
        },
        {
            "exito": False,
            "latencia_ms": 20,
            "palabras_clave_requeridas": 1,
            "claves_ok": False,
            "fuentes_requeridas": 1,
            "fuentes_ok": False,
            "debe_abstenerse": True,
            "abstencion_ok": False,
        },
        {
            "exito": True,
            "latencia_ms": 30,
            "palabras_clave_requeridas": 0,
            "claves_ok": True,
            "fuentes_requeridas": 0,
            "fuentes_ok": True,
            "debe_abstenerse": False,
            "abstencion_ok": True,
        },
    ]

    metricas = resumen(resultados)

    assert metricas["tasa_exito"] == 66.67
    assert (metricas["palabras_clave_ok"], metricas["palabras_clave_casos"]) == (1, 2)
    assert metricas["tasa_palabras_clave"] == 50.0
    assert (metricas["fuentes_requeridas_ok"], metricas["fuentes_requeridas_casos"]) == (
        1,
        2,
    )
    assert metricas["tasa_fuentes_requeridas"] == 50.0
    assert (metricas["abstenciones_ok"], metricas["abstenciones_requeridas"]) == (0, 1)
    assert metricas["tasa_abstenciones"] == 0.0


def test_resumen_informa_n_a_si_no_hay_criterios_aplicables():
    metricas = resumen([])

    assert metricas["tasa_palabras_clave"] is None
    assert metricas["tasa_fuentes_requeridas"] is None
    assert metricas["tasa_abstenciones"] is None
