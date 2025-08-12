#!/bin/bash
set -e

echo "⬆️ Actualizando pip..."
pip install --upgrade pip

echo "📦 Instalando PyTorch CPU-only (v2.1.0)..."
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "📦 Instalando dependencias..."
pip install -r requirements.txt

echo "✅ Instalación completada."

