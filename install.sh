#!/bin/bash
set -e

echo "⬆️ Actualizando pip..."
pip install --upgrade pip

echo "📦 Instalando PyTorch CPU-only..."
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo "📦 Instalando dependencias..."
pip install -r requirements.txt

echo "✅ Instalación completada."
