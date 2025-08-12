#!/bin/bash

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch CPU-only con la URL oficial de ruedas precompiladas
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Instalar transformers y otros paquetes
pip install transformers ftfy pandas openpyxl

# Si tienes otros paquetes, añádelos aquí, ejemplo:
# pip install flask flask-cors
