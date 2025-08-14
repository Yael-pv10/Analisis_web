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
    print("🔹 Insertando resultados de análisis en transcriptions.csv...")
    
    # Verificar que existe el archivo
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'Archivo transcriptions.csv no encontrado'}), 400
    
    try:
        # 1. LEER el archivo original
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        print(f"📂 Archivo leído: {len(df)} filas")
        
        # 2. PROCESAR cada transcripción si no está procesada
        print("⚡ Analizando sentimientos...")
        
        resultados = []
        for index, row in df.iterrows():
            texto = row.get('transcription', '')
            
            # Procesar si hay texto
            if pd.notna(texto) and texto.strip():
                resultado = analyze_sentiment_vader(texto.strip())
                resultados.append({
                    'sentimiento_predicho': resultado['label'],
                    'confianza': resultado['score'],
                    'rank_sentimiento': resultado['rank']
                })
            else:
                resultados.append({
                    'sentimiento_predicho': 'Sin texto',
                    'confianza': 0.0,
                    'rank_sentimiento': 0
                })
        
        # 3. AÑADIR las nuevas columnas al DataFrame original
        df['sentimiento_predicho'] = [r['sentimiento_predicho'] for r in resultados]
        df['confianza'] = [r['confianza'] for r in resultados]
        df['rank_sentimiento'] = [r['rank_sentimiento'] for r in resultados]
        
        print(f"✅ Añadidas 3 columnas nuevas")
        
        # 4. GUARDAR el archivo con las columnas nuevas
        df.to_csv(TRANSCRIPTIONS_CSV, index=False, encoding='utf-8')
        print(f"💾 Archivo actualizado: {TRANSCRIPTIONS_CSV}")
        
        # 5. VERIFICAR que se guardó correctamente
        df_verificacion = pd.read_csv(TRANSCRIPTIONS_CSV)
        columnas_nuevas = ['sentimiento_predicho', 'confianza', 'rank_sentimiento']
        
        if all(col in df_verificacion.columns for col in columnas_nuevas):
            print("✅ Verificación exitosa - Columnas añadidas correctamente")
            
            # Contar resultados
            positivos = len(df_verificacion[df_verificacion['sentimiento_predicho'] == 'Positivo'])
            negativos = len(df_verificacion[df_verificacion['sentimiento_predicho'] == 'Negativo'])
            neutrales = len(df_verificacion[df_verificacion['sentimiento_predicho'] == 'Neutral'])
            
            print(f"📊 Resultados: {positivos} Positivos, {negativos} Negativos, {neutrales} Neutrales")
            
            # Preparar respuesta
            response = {
                'success': True,
                'message': 'Análisis completado y guardado en transcriptions.csv',
                'total_transcriptions': len(df_verificacion),
                'results': {
                    'positivos': positivos,
                    'negativos': negativos,
                    'neutrales': neutrales
                },
                'columns_added': columnas_nuevas,
                'processed_opinions': df_verificacion.to_dict(orient='records')
            }
            
            return jsonify(response), 200
        else:
            return jsonify({'error': 'Error: Las columnas no se guardaron correctamente'}), 500
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


# Endpoint simple para verificar las columnas añadidas
@app.route('/check_columns', methods=['GET'])
def check_columns():
    """Verificar qué columnas tiene el archivo transcriptions.csv"""
    
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'Archivo no encontrado'}), 404
    
    try:
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        
        info = {
            'total_filas': len(df),
            'columnas_actuales': list(df.columns),
            'tiene_sentimientos': 'sentimiento_predicho' in df.columns
        }
        
        # Si ya tiene las columnas de sentimiento, mostrar distribución
        if info['tiene_sentimientos']:
            distribución = df['sentimiento_predicho'].value_counts().to_dict()
            info['distribución_sentimientos'] = distribución
            
            # Mostrar algunos ejemplos
            ejemplos = df[['transcription', 'sentimiento_predicho', 'confianza']].head(5)
            info['ejemplos'] = ejemplos.to_dict(orient='records')
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint para ver el archivo completo con las nuevas columnas
@app.route('/show_updated_file', methods=['GET'])
def show_updated_file():
    """Mostrar el archivo transcriptions.csv con todas sus columnas"""
    
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'Archivo no encontrado'}), 404
    
    try:
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        
        return jsonify({
            'archivo': 'transcriptions.csv',
            'total_filas': len(df),
            'columnas': list(df.columns),
            'datos': df.to_dict(orient='records')
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FUNCIÓN PARA VERIFICAR QUE LOS DATOS ESTÁN EN EL ARCHIVO ORIGINAL
@app.route('/verify_original_file', methods=['GET'])
def verify_original_file():
    """Verificar que los datos están en el archivo transcriptions.csv original"""
    try:
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            return jsonify({'error': 'Archivo original no encontrado'}), 404
        
        # Leer archivo original
        df = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
        
        # Análisis completo
        analysis = {
            'file_path': os.path.abspath(TRANSCRIPTIONS_CSV),
            'total_rows': len(df),
            'columns': list(df.columns),
            'has_sentiment_columns': all(col in df.columns for col in ['sentimiento_predicho', 'score', 'rank']),
            'file_size_bytes': os.path.getsize(TRANSCRIPTIONS_CSV),
            'last_modified': time.ctime(os.path.getmtime(TRANSCRIPTIONS_CSV))
        }
        
        if analysis['has_sentiment_columns']:
            # Contar sentimientos
            sentiment_counts = df['sentimiento_predicho'].value_counts().to_dict()
            processed_count = len(df[df['sentimiento_predicho'] != 'Sin procesar'])
            
            analysis.update({
                'sentiment_distribution': sentiment_counts,
                'processed_count': processed_count,
                'unprocessed_count': len(df) - processed_count,
                'processing_percentage': round((processed_count / len(df)) * 100, 2) if len(df) > 0 else 0
            })
            
            # Mostrar ejemplos de datos procesados
            processed_examples = df[df['sentimiento_predicho'] != 'Sin procesar'].head(3)
            analysis['sample_processed_data'] = processed_examples.to_dict(orient='records')
        
        return jsonify(analysis), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# FUNCIÓN PARA MOSTRAR CONTENIDO COMPLETO DEL ARCHIVO ORIGINAL
@app.route('/show_original_content', methods=['GET'])
def show_original_content():
    """Mostrar todo el contenido del archivo transcriptions.csv original"""
    try:
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            return jsonify({'error': 'Archivo original no encontrado'}), 404
        
        df = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
        
        return jsonify({
            'file': 'transcriptions.csv (ORIGINAL)',
            'total_rows': len(df),
            'columns': list(df.columns),
            'data': df.to_dict(orient='records')
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ENDPOINT PARA COMPARAR ARCHIVOS (SI EXISTEN AMBOS)
@app.route('/compare_files', methods=['GET'])
def compare_files():
    """Comparar el archivo original con cualquier archivo procesado"""
    try:
        files_info = {}
        
        # Verificar archivo original
        if os.path.exists(TRANSCRIPTIONS_CSV):
            df_original = pd.read_csv(TRANSCRIPTIONS_CSV, encoding='utf-8-sig')
            files_info['original'] = {
                'file': TRANSCRIPTIONS_CSV,
                'exists': True,
                'rows': len(df_original),
                'columns': list(df_original.columns),
                'has_sentiment': 'sentimiento_predicho' in df_original.columns,
            }
            
            if files_info['original']['has_sentiment']:
                processed = len(df_original[df_original['sentimiento_predicho'] != 'Sin procesar'])
                files_info['original']['processed_count'] = processed
        
        # Verificar archivos adicionales
        additional_files = ['opiniones_con_sentimientos.csv', 'processed_transcriptions.csv']
        for file_name in additional_files:
            if os.path.exists(file_name):
                df_additional = pd.read_csv(file_name, encoding='utf-8-sig')
                files_info[file_name] = {
                    'file': file_name,
                    'exists': True,
                    'rows': len(df_additional),
                    'columns': list(df_additional.columns),
                }
        
        return jsonify(files_info), 200
        
    except Exception as e:
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