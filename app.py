from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import pandas as pd
import os
from flask_cors import CORS
from pydub import AudioSegment
import uuid
from ftfy import fix_text
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
from celery import Celery
import time

# Configuración de Celery para procesamiento asíncrono
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
CORS(app)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo ligero (25x más rápido que BERT)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Modelo optimizado para CPU
MODEL_CACHE = {}

def load_model():
    if "model" not in MODEL_CACHE:
        logger.info("Cargando modelo ligero...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            MODEL_CACHE["model"] = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1  # Forzar CPU
            )
            logger.info("Modelo ligero cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            MODEL_CACHE["model"] = None
    return MODEL_CACHE["model"]

classifier = load_model()

# Configuraciones
AUDIO_DIR = 'audios'
os.makedirs(AUDIO_DIR, exist_ok=True)

# Mapeo de sentimientos
SENTIMENT_MAP = {
    "POSITIVE": "Positivo",
    "NEGATIVE": "Negativo",
    "NEUTRAL": "Neutral"
}

@celery.task(bind=True)
def async_analyze_batch(self, textos):
    """Tarea Celery para análisis asíncrono"""
    classifier = load_model()
    if not classifier:
        return {"status": "error", "error": "Modelo no disponible"}
    
    total = len(textos)
    resultados = []
    
    for i, texto in enumerate(textos):
        try:
            if pd.isna(texto) or str(texto).strip() == "":
                resultados.append({
                    "label": "No disponible",
                    "score": 0,
                    "sentimiento": "No disponible"
                })
                continue
            
            prediction = classifier(str(texto))[0]
            resultados.append({
                "label": prediction["label"],
                "score": prediction["score"],
                "sentimiento": SENTIMENT_MAP.get(prediction["label"], prediction["label"])
            })
            
            # Actualizar progreso (0-100)
            self.update_state(state='PROGRESS',
                            meta={'current': i+1, 
                                 'total': total,
                                 'progress': int(((i+1)/total)*100)})
            
            # Pequeña pausa para CPU
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error procesando texto: {e}")
            resultados.append({
                "label": "Error",
                "score": 0,
                "sentimiento": "Error"
            })
    
    return {
        "status": "completed",
        "results": resultados,
        "progress": 100
    }

@app.route('/start_async_analysis', methods=['POST'])
def start_async_analysis():
    """Endpoint para iniciar análisis asíncrono"""
    if not os.path.exists('transcriptions.csv'):
        return jsonify({"error": "No hay transcripciones para analizar"}), 400
    
    df = pd.read_csv('transcriptions.csv')
    textos = df['transcription'].apply(lambda x: fix_text(str(x)) if pd.notna(x) else x).tolist()
    
    # Iniciar tarea asíncrona
    task = async_analyze_batch.apply_async(args=[textos])
    
    return jsonify({
        "task_id": task.id,
        "status": "PENDING",
        "progress": 0
    }), 202

@app.route('/check_status/<task_id>', methods=['GET'])
def check_status(task_id):
    """Endpoint para verificar estado de tarea"""
    task = async_analyze_batch.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'status': 'PENDING',
            'progress': 0
        }
    elif task.state == 'PROGRESS':
        response = {
            'status': 'PROGRESS',
            'progress': task.info.get('progress', 0),
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1)
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': 'COMPLETED',
            'progress': 100,
            'results': task.info.get('results', [])
        }
    else:
        response = {
            'status': 'FAILED',
            'error': str(task.info)
        }
    
    return jsonify(response)

# Resto de endpoints (upload, transcribe, etc... se mantienen igual)
# ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
