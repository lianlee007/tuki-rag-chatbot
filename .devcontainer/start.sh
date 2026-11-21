#!/usr/bin/env bash
# Arranca Ollama, asegura los modelos y levanta la API en el puerto 8000.
set -uo pipefail

cd "$(dirname "$0")/.."

if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
fi

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text

if [ -f Modelfile ]; then
  ollama create tuki-chef -f Modelfile || true
fi

pkill -f "uvicorn app.main:app" 2>/dev/null || true

OLLAMA_URL=http://127.0.0.1:11434 \
CATALOGO_PATH=datos/catalogo.json \
OLLAMA_MODELO_CHAT=tuki-chef \
OLLAMA_MODELO_EMBEDDINGS=nomic-embed-text \
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &

echo "API arrancada en el puerto 8000."
