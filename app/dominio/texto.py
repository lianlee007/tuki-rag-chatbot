"""Utilidades de texto para el RAG local.

No dependen de Ollama ni de FastAPI. Se usan para normalizar la pregunta, para
construir el índice BM25 y para partir descripciones largas en fragmentos con
solapamiento.
"""

from __future__ import annotations

import re
import unicodedata

# 1 palabra ≈ 1.33 tokens para el español. Es una estimación deliberada:
# añadir un tokenizador real costaría una dependencia y varios MB de RAM, y
# para dimensionar el contexto de un modelo de 1.5B la diferencia no importa.
TOKENS_POR_PALABRA = 1.33

_STOPWORDS = {
    "a", "al", "algo", "algun", "alguna", "algunas", "alguno", "algunos",
    "ante", "antes", "como", "con", "contra", "cual", "cuales", "cuando",
    "de", "del", "desde", "donde", "dos", "el", "ella", "ellas", "ellos",
    "en", "entre", "era", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "estas", "este", "esto", "estos", "favor", "fue", "gracias",
    "ha", "hasta", "hay", "hola", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "mis", "mucho", "muy", "no", "nos", "o", "otra",
    "otro", "para", "pero", "poco", "por", "porque", "que", "quien",
    "quiere", "quiero", "se", "ser", "si", "sin", "sobre", "su", "sus",
    "te", "tiene", "tienes", "todo", "tu", "tus", "un", "una", "uno",
    "unos", "y", "ya", "yo", "recomienda", "recomiendas", "recomendarme",
    "sugiere", "sugiereme", "busco", "necesito", "puedes", "podrias",
}

_GRUPOS_SINONIMOS = (
    {"barato", "barata", "economic", "economica", "asequible", "presupuesto"},
    {"vegetariano", "vegetariana", "veggie", "vegetal", "sin", "carne"},
    {"vegano", "vegana", "vegan", "plant", "based"},
    {"picante", "picoso", "picosito", "aji", "chile", "guayaba"},
    {"postre", "dulce", "torta", "helado", "reposteria"},
    {"bebida", "jugo", "cafe", "refresco", "limonada", "cerveza"},
    {"desayuno", "desayunar", "manana", "huevos", "calentado", "cafe", "torta",
     "arepa", "envuelto", "mazamorra"},
    {"almuerzo", "almorzar", "corrientazo", "menu", "mediodia"},
    {"cena", "cenar", "noche"},
    {"bebida", "tomar", "jugo", "limonada", "cafe", "refresco", "cerveza"},
    {"compartir", "pizza", "picada", "parrillada", "dos"},
    {"acompanar", "torta", "postre", "bebida", "cafe"},
    {"contundente", "pizza", "bandeja", "churrasco", "parrillada", "carne"},
    {"pizza", "pizzas", "italiana", "mozzarella"},
    {"hamburguesa", "hamburguesas", "burger"},
    {"pollo", "pollos", "asado", "parrilla", "carne"},
    {"pescado", "mariscos", "camaron", "sushi"},
    {"rapido", "domicilio", "delivery", "pedir"},
    {"sano", "saludable", "ligero", "fresco", "ensalada"},
)

PALABRAS_GASTRONOMICAS = {
    "comer", "comida", "plato", "platos", "restaurante", "restaurantes",
    "menu", "carta", "almorzar", "almuerzo", "cenar", "cena", "desayunar",
    "desayuno", "pedir", "domicilio", "delivery", "recomienda", "recomiendas",
    "recomendacion", "sugerencia", "hambre", "antojo", "probar", "degustar",
    "vegetariano", "vegano", "picante", "postre", "bebida", "cafe", "pizza",
    "hamburguesa", "pollo", "carne", "pescado", "mariscos", "sushi", "ensalada",
    "barato", "economico", "precio", "presupuesto", "reserva", "mesa",
    "cocina", "chef", "ingrediente", "ingredientes", "alergia", "alergeno",
    "gluten", "lactosa", "sin", "veggie", "corrientazo", "arepa",
    "lleva", "llevan", "sabor", "porcion",
    "tamano", "dulce", "salado", "acido", "suave", "fuerte", "caliente",
    "frio", "cuesta", "vale", "horario", "abierto", "cierra", "cierre",
    "direccion", "queda", "ubicado", "telefono", "whatsapp", "torta",
    "capuchino", "patacones", "mazamorra", "obleas", "envueltos", "salpicon",
    "lasagna", "pasta", "bandeja", "sancocho", "ajiaco", "tamal",
    "empanada", "churrasco", "costilla", "cerdo", "res", "pizzeria",
    "panaderia", "cafeteria", "vegetarianos", "veganos", "picantes",
    "porciones", "bebidas", "postres", "desayunos", "almuerzos",
    "cenas", "sopa", "crema", "jugo", "limonada", "cerveza",
    "vegetariana", "vegetarianas", "vegana", "veganas", "economica",
    "economicas", "economicos", "tomar", "compartir",
    "acompanar", "contundente", "recomiendame", "recomendame", "rico",
    "sabroso", "delicioso", "opcion", "opciones", "menos", "maximo",
    "hasta", "mil", "lacteos", "lacteo", "alergias",
}

_TEMAS_NO_GASTRONOMICOS = {
    "politica", "presidente", "elecciones", "medicina", "medico", "enfermedad",
    "receta medica", "programacion", "python", "javascript", "futbol",
    "partido", "impuestos", "abogado", "demanda", "guerra", "religion",
}

_SALIDA_CODIGO = re.compile(r"```[a-z]*\s*|```", re.IGNORECASE)
_SALIDA_ETIQUETA = re.compile(
    r"^\s*(eslogan|descripci[oó]n|nombre|categor[ií]a|ingredientes|respuesta|resultado)"
    r"\s*:\s*",
    re.IGNORECASE,
)
_COMILLAS = "\"'“”‘’«»"


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y con espacios simples."""
    sin_acentos = unicodedata.normalize("NFKD", texto or "")
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    limpio = re.sub(r"[^a-z0-9ñ]+", " ", sin_acentos.lower())
    return re.sub(r"\s+", " ", limpio).strip()


def tokenizar(texto: str, *, quitar_stopwords: bool = True) -> list[str]:
    """Tokens normalizados con una stemming muy ligera de plurales."""
    tokens: list[str] = []
    for bruto in normalizar(texto).split():
        if len(bruto) <= 1:
            continue
        token = bruto[:-1] if len(bruto) > 3 and bruto.endswith("s") else bruto
        if quitar_stopwords and token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def expandir_consulta(tokens: list[str]) -> list[tuple[str, float]]:
    """Añade sinónimos con menos peso para no diluir la coincidencia exacta."""
    pesos: dict[str, float] = {token: 1.0 for token in tokens}
    conjunto = set(tokens)
    for grupo in _GRUPOS_SINONIMOS:
        if conjunto & grupo:
            for sinonimo in grupo:
                if sinonimo not in pesos and sinonimo not in _STOPWORDS:
                    pesos[sinonimo] = 0.35
    return list(pesos.items())


def estimar_tokens(texto: str) -> int:
    palabras = len(texto.split())
    return max(1, int(round(palabras * TOKENS_POR_PALABRA)))


def chunk_texto(texto: str, max_tokens: int = 220, overlap: int = 40) -> list[str]:
    """Parte un texto largo en fragmentos de ``max_tokens`` con solapamiento.

    Los platos y los perfiles cortos no se tocan. Solo las descripciones
    largas se parten, y se hace por palabras para que el corte sea barato y
    reproducible.
    """
    limpio = re.sub(r"\s+", " ", texto or "").strip()
    if not limpio:
        return []
    if estimar_tokens(limpio) <= max_tokens:
        return [limpio]

    max_palabras = max(8, int(max_tokens / TOKENS_POR_PALABRA))
    solape = max(0, min(max_palabras - 1, int(overlap / TOKENS_POR_PALABRA)))
    paso = max(1, max_palabras - solape)
    palabras = limpio.split()
    fragmentos: list[str] = []
    for inicio in range(0, len(palabras), paso):
        trozo = palabras[inicio : inicio + max_palabras]
        if not trozo:
            break
        fragmentos.append(" ".join(trozo))
        if inicio + max_palabras >= len(palabras):
            break
    return fragmentos


def es_consulta_gastronomica(texto: str) -> bool:
    """Distingue una pregunta de comida de un tema ajeno al restaurante."""
    normalizado = normalizar(texto)
    if not normalizado:
        return False
    if any(tema in normalizado for tema in _TEMAS_NO_GASTRONOMICOS):
        # Un tema ajeno solo pasa si además hay una palabra claramente de comida.
        return any(palabra in normalizado.split() for palabra in PALABRAS_GASTRONOMICAS)
    palabras = set(normalizado.split())
    return bool(palabras & PALABRAS_GASTRONOMICAS)


def limpiar_salida(texto: str) -> str:
    """Quita bloques de código, etiquetas y comillas que añaden los modelos."""
    texto = _SALIDA_CODIGO.sub("", texto or "")
    texto = _SALIDA_ETIQUETA.sub("", texto)
    return texto.strip().strip(_COMILLAS).strip()
