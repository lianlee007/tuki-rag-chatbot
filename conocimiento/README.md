# Base de conocimiento

El chatbot no usa PDFs sueltos. Usa el catálogo estructurado de TukiTuki:

- `datos/catalogo.json`: 7 restaurantes y 61 platos exportados de PostgreSQL.
- `conocimiento/catalogo.md`: la misma información en un documento legible.
- Cada plato incluye nombre, restaurante, ciudad, categoría, precio, descripción,
  ingredientes, alérgenos y banderas de vegetariano/vegano/picante/destacado.
- Cada restaurante incluye ciudad, cocinas, horario, dirección, teléfono y
  valoración media.

## Documentos que se indexan

1. **Plato**: un documento atómico por plato. Así el precio y los alérgenos no se
   separan de su descripción.
2. **Restaurante**: un documento por ficha de restaurante.
3. **Descripciones largas**: se parten en fragmentos.

## Corte y solapamiento

- Tamaño de fragmento: **220 tokens**.
- Solapamiento: **40 tokens** (18 %).
- Los platos y las fichas cortas no se parten.
- Los tokens se estiman con una heurística local: 1 palabra ≈ 1.33 tokens.
- No se usa un tokenizador externo para mantener el repo liviano.

## Recuperación

- BM25 en Python puro para nombres, ingredientes y precios.
- Embeddings de `nomic-embed-text` para preguntas semánticas.
- Fusión con Reciprocal Rank Fusion (k=60).
- Filtros por dieta, precio, ciudad, restaurante y alérgenos.
- Selección final con MMR (λ=0.72).

## Cómo se regenera

```bash
python scripts/exportar_catalogo.py
python scripts/generar_markdown.py
```

El primer script necesita que `restaurantes-service` esté corriendo en
`http://localhost:3002`. El segundo solo lee `datos/catalogo.json`.
