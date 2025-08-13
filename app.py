from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import pandas as pd
import os
from flask_cors import CORS
from pydub import AudioSegment
import uuid
from ftfy import fix_text
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Importar modelo más rápido
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Alternativa con transformers más ligero (descomenta si prefieres)
# from transformers import pipeline

# Inicializar analizador VADER (muy rápido)
analyzer = SentimentIntensityAnalyzer()

# Si prefieres usar un modelo de transformers más ligero, usa esto:
# classifier = pipeline("sentiment-analysis", 
#                      model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#                      device=-1)  # Usar CPU

# Preprocesamiento más simple
import nltk
import string
from collections import Counter

# Descargas necesarias de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configuración del servidor
app = Flask(__name__)
CORS(app)

AUDIO_DIR = 'audios'
TRANSCRIPTIONS_CSV = 'transcriptions.csv'
PROCESSED_CSV = 'processed_transcriptions.csv'
os.makedirs(AUDIO_DIR, exist_ok=True)

# Palabras clave para español
stop_words_spanish = set([
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 
    'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'pero', 'está', 'muy', 'más', 'como', 'me', 'ya'
])

# Cache para resultados
sentiment_cache = {}
cache_lock = threading.Lock()

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    original_ext = file.filename.split('.')[-1].lower()
    unique_name = f"{uuid.uuid4()}.{original_ext}"
    audio_path = os.path.join(AUDIO_DIR, unique_name)
    file.save(audio_path)

    # Convertir webm/mp3 a wav
    if original_ext in ['webm', 'mp3']:
        audio_wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        try:
            sound = AudioSegment.from_file(audio_path)
            sound.export(audio_wav_path, format='wav')
            audio_path = audio_wav_path
        except Exception as e:
            return jsonify({'error': f'Error al convertir el audio: {e}'}), 500

    transcription = transcribe_audio(audio_path)
    save_transcription_to_csv(file.filename, transcription)

    return jsonify({'transcription': transcription}), 200

def preprocess_text_fast(text):
    """Preprocesamiento rápido y simple"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower().strip()
    
    # Eliminar URLs, menciones, hashtags
    text = re.sub(r'http\S+|www.\S+|@\w+|#\w+', '', text)
    
    # Eliminar caracteres especiales pero mantener emojis básicos
    text = re.sub(r'[^\w\s!?¡¿.,;:()]', '', text)
    
    # Limpiar espacios múltiples
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment_vader(text):
    """Análisis de sentimiento con VADER (muy rápido)"""
    if not text or text.strip() == "":
        return {"label": "Neutral", "score": 0.0, "rank": 2}
    
    # Verificar cache
    with cache_lock:
        if text in sentiment_cache:
            return sentiment_cache[text]
    
    try:
        # VADER análisis
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Clasificación más estricta (solo positivo/negativo)
        if compound >= 0.1:
            label = "Positivo"
            rank = 3
        elif compound <= -0.1:
            label = "Negativo"  
            rank = 1
        else:
            label = "Neutral"
            rank = 2
            
        result = {
            "label": label,
            "score": abs(compound),
            "rank": rank
        }
        
        # Guardar en cache
        with cache_lock:
            sentiment_cache[text] = result
            
        return result
        
    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return {"label": "Neutral", "score": 0.0, "rank": 2}

def analyze_sentiment_batch(texts):
    """Procesar múltiples textos en paralelo"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(analyze_sentiment_vader, texts))
    return results

@app.route('/analyze_sentiments', methods=['POST'])
def analyze_sentiments():
    print("🔹 Iniciando análisis de sentimientos optimizado...")
    start_time = time.time()

    # Verificar si el archivo CSV existe
    file_path = "transcriptions.csv"
    if not os.path.exists(file_path):
        print(f"❌ No se encontró el archivo: {file_path}")
        return jsonify({'error': 'Archivo CSV no encontrado'}), 400

    try:
        print(f"📂 Leyendo archivo: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Archivo leído correctamente. {len(df)} filas encontradas")

        # Verificar columna transcription
        if "transcription" not in df.columns:
            print("❌ La columna 'transcription' no existe en el CSV")
            return jsonify({'error': "Columna 'transcription' no encontrada"}), 400

        # Preprocesar texto rápidamente
        print("🔹 Preprocesando texto...")
        df["transcription_clean"] = df["transcription"].apply(preprocess_text_fast)
        
        # Filtrar textos vacíos
        df_clean = df[df["transcription_clean"].str.len() > 0].copy()
        print(f"📊 {len(df_clean)} textos válidos para procesar")

        # Análisis de sentimiento en lotes
        print("🔹 Aplicando análisis de sentimientos...")
        texts = df_clean["transcription_clean"].tolist()
        
        # Procesar en lotes pequeños para mejor rendimiento
        batch_size = 50
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = analyze_sentiment_batch(batch)
            all_results.extend(batch_results)
            
            # Mostrar progreso
            processed = min(i + batch_size, len(texts))
            print(f"   Procesado: {processed}/{len(texts)} ({processed/len(texts)*100:.1f}%)")

        # Asignar resultados
        df_clean["sentimiento_predicho"] = [r["label"] for r in all_results]
        df_clean["score"] = [r["score"] for r in all_results]
        df_clean["rank"] = [r["rank"] for r in all_results]

        # Filtrar solo positivos y negativos (omitir neutrales)
        df_filtered = df_clean[df_clean["sentimiento_predicho"].isin(["Positivo", "Negativo"])].copy()
        print(f"📊 Resultados filtrados: {len(df_filtered)} opiniones (solo positivas/negativas)")

        # Guardar CSV con resultados
        output_path = "opiniones_con_sentimientos.csv"
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # Preparar respuesta con estructura esperada por el frontend
        response_data = []
        for _, row in df_filtered.iterrows():
            response_data.append({
                "transcription": row["transcription"],
                "sentimiento_predicho": row["sentimiento_predicho"],
                "rank": row["rank"],
                "score": round(row["score"], 3)
            })

        processing_time = time.time() - start_time
        print(f"✅ Procesamiento completado en {processing_time:.2f} segundos")
        print(f"📄 Resultados guardados en: {output_path}")

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error en analyze_sentiments: {e}")
        return jsonify({'error': str(e)}), 500

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language='es-ES')
        except sr.UnknownValueError:
            return "No se pudo entender el audio"
        except sr.RequestError as e:
            return f"Error en el servicio de reconocimiento: {e}"

def save_transcription_to_csv(filename, transcription):
    df = pd.DataFrame([[filename, transcription]], columns=['filename', 'transcription'])
    if not os.path.isfile(TRANSCRIPTIONS_CSV):
        df.to_csv(TRANSCRIPTIONS_CSV, index=False)
    else:
        df.to_csv(TRANSCRIPTIONS_CSV, mode='a', header=False, index=False)

@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    transcription = request.form.get('transcription')
    filename = request.form.get('filename', 'transcripcion_manual')
    if transcription:
        save_transcription_to_csv(filename, transcription)
        return jsonify({'message': 'Transcripción guardada exitosamente'}), 200
    return jsonify({'error': 'No se recibió transcripción'}), 400

@app.route('/get_transcriptions', methods=['GET'])
def get_transcriptions():
    """Endpoint para obtener todas las transcripciones"""
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'No hay archivo de transcripciones'}), 404
    
    try:
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud para verificar que el servidor funciona"""
    return jsonify({
        'status': 'healthy',
        'model': 'VADER Sentiment',
        'cache_size': len(sentiment_cache)
    }), 200

@app.route('/download_processed_csv', methods=['GET'])
def download_processed_csv():
    output_file = "opiniones_con_sentimientos.csv"
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return jsonify({'error': 'Archivo procesado no encontrado'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Iniciando servidor en puerto {port}")
    print("📊 Usando VADER Sentiment Analysis para mejor rendimiento")
    app.run(host='0.0.0.0', port=port, debug=False)