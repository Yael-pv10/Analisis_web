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
    file_path = TRANSCRIPTIONS_CSV  # Usar la variable global definida
    if not os.path.exists(file_path):
        print(f"❌ No se encontró el archivo: {file_path}")
        return jsonify({'error': 'Archivo CSV no encontrado'}), 400

    try:
        print(f"📂 Leyendo archivo: {file_path}")
        # Leer el archivo con encoding correcto
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✅ Archivo leído correctamente. {len(df)} filas encontradas")

        # Verificar columna transcription
        if "transcription" not in df.columns:
            print("❌ La columna 'transcription' no existe en el CSV")
            return jsonify({'error': "Columna 'transcription' no encontrada"}), 400

        # Inicializar columnas de sentimiento si no existen
        if "sentimiento_predicho" not in df.columns:
            df["sentimiento_predicho"] = "Sin procesar"
        if "score" not in df.columns:
            df["score"] = 0.0
        if "rank" not in df.columns:
            df["rank"] = 0

        # Preprocesar texto rápidamente
        print("🔹 Preprocesando texto...")
        df["transcription_clean"] = df["transcription"].apply(preprocess_text_fast)
        
        # Identificar qué textos necesitan procesarse
        mask_to_process = (
            (df["transcription_clean"].str.len() > 0) & 
            (df["sentimiento_predicho"] == "Sin procesar")
        )
        
        df_to_process = df[mask_to_process].copy()
        print(f"📊 {len(df_to_process)} textos nuevos para procesar")
        print(f"📊 {len(df[df['sentimiento_predicho'] != 'Sin procesar'])} ya procesados anteriormente")

        if len(df_to_process) == 0:
            print("ℹ️ No hay textos nuevos para procesar")
            # Devolver datos existentes
            processed_data = df[df["sentimiento_predicho"] != "Sin procesar"].copy()
            response_data = {
                "total_transcriptions": len(df),
                "processed_opinions": []
            }
            
            for _, row in processed_data.iterrows():
                response_data["processed_opinions"].append({
                    "transcription": row["transcription"],
                    "sentimiento_predicho": row["sentimiento_predicho"],
                    "rank": row["rank"],
                    "score": round(float(row["score"]), 3)
                })
            
            return jsonify(response_data), 200

        # Análisis de sentimiento en lotes para textos nuevos
        print("🔹 Aplicando análisis de sentimientos...")
        texts_to_process = df_to_process["transcription_clean"].tolist()
        
        # Procesar en lotes pequeños para mejor rendimiento
        batch_size = 50
        all_results = []
        
        for i in range(0, len(texts_to_process), batch_size):
            batch = texts_to_process[i:i+batch_size]
            batch_results = analyze_sentiment_batch(batch)
            all_results.extend(batch_results)
            
            # Mostrar progreso
            processed = min(i + batch_size, len(texts_to_process))
            print(f"   Procesado: {processed}/{len(texts_to_process)} ({processed/len(texts_to_process)*100:.1f}%)")

        # Asignar resultados al DataFrame original
        print("🔹 Guardando resultados...")
        processed_indices = df_to_process.index.tolist()
        
        for i, result in enumerate(all_results):
            idx = processed_indices[i]
            df.loc[idx, "sentimiento_predicho"] = result["label"]
            df.loc[idx, "score"] = result["score"]
            df.loc[idx, "rank"] = result["rank"]

        # Eliminar columna temporal de texto limpio antes de guardar
        if "transcription_clean" in df.columns:
            df = df.drop("transcription_clean", axis=1)

        # Guardar el archivo original actualizado
        try:
            # Hacer backup del archivo original por seguridad
            backup_path = f"{TRANSCRIPTIONS_CSV}.backup"
            if os.path.exists(TRANSCRIPTIONS_CSV):
                import shutil
                shutil.copy2(TRANSCRIPTIONS_CSV, backup_path)
                print(f"📋 Backup creado: {backup_path}")
            
            # Guardar archivo actualizado
            df.to_csv(TRANSCRIPTIONS_CSV, index=False, encoding='utf-8-sig')
            print(f"✅ Archivo original actualizado: {TRANSCRIPTIONS_CSV}")
            
            # Verificar que se guardó correctamente
            df_verify = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
            processed_count = len(df_verify[df_verify["sentimiento_predicho"] != "Sin procesar"])
            print(f"📊 Verificación: {processed_count} registros procesados guardados")
            
        except Exception as save_error:
            print(f"❌ Error al guardar archivo: {save_error}")
            return jsonify({'error': f'Error al guardar: {save_error}'}), 500

        # Preparar respuesta con TODOS los datos procesados
        response_data = {
            "total_transcriptions": len(df),
            "processed_opinions": []
        }
        
        # Solo enviar los datos que tienen sentimiento procesado
        df_processed = df[df["sentimiento_predicho"] != "Sin procesar"].copy()
        
        for _, row in df_processed.iterrows():
            response_data["processed_opinions"].append({
                "transcription": str(row["transcription"]),
                "sentimiento_predicho": str(row["sentimiento_predicho"]),
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else 0,
                "score": round(float(row["score"]), 3) if pd.notna(row["score"]) else 0.0
            })

        processing_time = time.time() - start_time
        print(f"✅ Procesamiento completado en {processing_time:.2f} segundos")
        print(f"📊 Total transcripciones: {len(df)}")
        print(f"📊 Nuevas procesadas: {len(all_results)}")
        print(f"📊 Total procesadas: {len(df_processed)}")

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error en analyze_sentiments: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# También actualiza el endpoint para verificar el estado del archivo
@app.route('/get_transcriptions', methods=['GET'])
def get_transcriptions():
    """Endpoint para obtener todas las transcripciones con sentimientos si existen"""
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'No hay archivo de transcripciones'}), 404
    
    try:
        df = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
        
        # Verificar si ya tiene columnas de sentimiento
        has_sentiment = 'sentimiento_predicho' in df.columns
        processed_count = 0
        
        if has_sentiment:
            # Contar cuántos tienen sentimiento procesado (no "Sin procesar")
            processed_count = len(df[
                (df['sentimiento_predicho'].notna()) & 
                (df['sentimiento_predicho'] != 'Sin procesar') &
                (df['sentimiento_predicho'] != '')
            ])
        
        return jsonify({
            'total_count': len(df),
            'processed_count': processed_count,
            'has_sentiment_data': has_sentiment,
            'transcriptions': df.to_dict(orient='records')
        }), 200
    except Exception as e:
        print(f"Error en get_transcriptions: {e}")
        return jsonify({'error': str(e)}), 500


# Función auxiliar para verificar la integridad del archivo
@app.route('/verify_csv', methods=['GET'])
def verify_csv():
    """Endpoint para verificar la integridad del archivo CSV"""
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'Archivo no encontrado'}), 404
    
    try:
        df = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
        
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'has_sentiment_columns': all(col in df.columns for col in ['sentimiento_predicho', 'score', 'rank']),
        }
        
        if stats['has_sentiment_columns']:
            stats['sentiment_stats'] = {
                'sin_procesar': len(df[df['sentimiento_predicho'] == 'Sin procesar']),
                'positivo': len(df[df['sentimiento_predicho'] == 'Positivo']),
                'negativo': len(df[df['sentimiento_predicho'] == 'Negativo']),
                'neutral': len(df[df['sentimiento_predicho'] == 'Neutral']),
            }
        
        return jsonify(stats), 200
    except Exception as e:
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