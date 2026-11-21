#!/usr/bin/env bash
# Prepara el Codespace: Ollama y dependencias de la API.
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-venv python3-pip curl ca-certificates

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Codspace preparado."
