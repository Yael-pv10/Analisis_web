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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json

# Agregar estas funciones al archivo app.py

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



def preprocess_for_ml(df):
    """Preprocesar datos para machine learning"""
    # Filtrar solo las filas que tienen sentimiento procesado
    df_processed = df[
        (df['sentimiento_predicho'].notna()) & 
        (df['sentimiento_predicho'] != 'Sin procesar') &
        (df['sentimiento_predicdo'] != '') &
        (df['transcription'].notna()) &
        (df['transcription'] != '')
    ].copy()
    
    # Limpiar textos
    df_processed['text_clean'] = df_processed['transcription'].apply(preprocess_text_fast)
    
    # Filtrar textos vacíos después del preprocesamiento
    df_processed = df_processed[df_processed['text_clean'].str.len() > 5]
    
    return df_processed

def create_confusion_matrix_plot(y_true, y_pred, labels):
    """Crear matriz de confusión como imagen base64"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión')
    plt.ylabel('Valores Reales')
    plt.xlabel('Predicciones')
    
    # Convertir a base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

@app.route('/train_model', methods=['POST'])
def train_model():
    """Entrenar modelo de clasificación con Hold-out"""
    try:
        # Verificar que existe el archivo
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            return jsonify({'error': 'Archivo transcriptions.csv no encontrado'}), 400
        
        # Cargar datos
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        
        # Verificar que hay datos con sentimientos
        if 'sentimiento_predicho' not in df.columns:
            return jsonify({'error': 'No hay datos de sentimiento. Ejecuta el análisis primero.'}), 400
        
        # Preprocesar datos
        df_processed = preprocess_for_ml(df)
        
        if len(df_processed) < 10:
            return jsonify({'error': f'Datos insuficientes para entrenamiento. Solo {len(df_processed)} registros válidos.'}), 400
        
        # Preparar características y etiquetas
        X = df_processed['text_clean'].values
        y = df_processed['sentimiento_predicho'].values
        
        # Vectorización TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(stop_words_spanish),
            ngram_range=(1, 2),
            min_df=2
        )
        
        X_tfidf = vectorizer.fit_transform(X)
        
        # Codificar etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # División Hold-out (80% entrenamiento, 20% prueba)
        test_size = float(request.json.get('test_size', 0.2))
        random_state = int(request.json.get('random_state', 42))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Entrenar modelo de Regresión Logística
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas de desempeño
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Reporte de clasificación
        class_names = label_encoder.classes_
        classification_rep = classification_report(
            y_test, y_pred_test, 
            target_names=class_names,
            output_dict=True
        )
        
        # Matriz de confusión
        confusion_img = create_confusion_matrix_plot(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred_test),
            class_names
        )
        
        # Distribución de clases
        class_distribution = pd.Series(y).value_counts().to_dict()
        
        # Características más importantes (coeficientes del modelo)
        feature_names = vectorizer.get_feature_names_out()
        top_features = {}
        
        for i, class_name in enumerate(class_names):
            if len(class_names) == 2:
                # Clasificación binaria
                coefficients = model.coef_[0] if i == 1 else -model.coef_[0]
            else:
                # Clasificación multiclase
                coefficients = model.coef_[i]
            
            top_indices = coefficients.argsort()[-10:][::-1]
            top_features[class_name] = [
                {
                    'feature': feature_names[idx],
                    'coefficient': float(coefficients[idx])
                }
                for idx in top_indices
            ]
        
        # Guardar resultados del modelo
        model_results = {
            'model_type': 'Logistic Regression',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_features': X_tfidf.shape[1],
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'class_distribution': class_distribution,
            'classification_report': classification_rep,
            'confusion_matrix_img': confusion_img,
            'top_features': top_features,
            'classes': list(class_names),
            'test_split_ratio': test_size,
            'random_state': random_state
        }
        
        # Guardar en archivo JSON para persistencia
        with open('model_results.json', 'w', encoding='utf-8') as f:
            # Crear una copia sin la imagen para el JSON
            json_results = model_results.copy()
            json_results.pop('confusion_matrix_img', None)
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Modelo entrenado exitosamente',
            'results': model_results
        }), 200
        
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_model_results', methods=['GET'])
def get_model_results():
    """Obtener resultados del último modelo entrenado"""
    try:
        # Intentar cargar desde archivo JSON
        if os.path.exists('model_results.json'):
            with open('model_results.json', 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Regenerar matriz de confusión si es necesario
            if 'confusion_matrix_img' not in results:
                results['confusion_matrix_img'] = None
            
            return jsonify({
                'success': True,
                'results': results
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'No hay resultados de modelo disponibles. Entrena un modelo primero.'
            }), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_ml():
    """Realizar predicción con el modelo entrenado"""
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No se proporcionó texto'}), 400
        
        # Cargar resultados del modelo (en un caso real, cargarías el modelo serializado)
        if not os.path.exists('model_results.json'):
            return jsonify({'error': 'No hay modelo entrenado disponible'}), 400
        
        # Por ahora, usar VADER como backup
        result = analyze_sentiment_vader(text)
        
        return jsonify({
            'text': text,
            'prediction': result['label'],
            'confidence': result['score'],
            'method': 'VADER (ML model not persisted)'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_statistics', methods=['GET'])
def model_statistics():
    """Obtener estadísticas del dataset para ML"""
    try:
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        
        # Estadísticas generales
        stats = {
            'total_records': len(df),
            'columns': list(df.columns)
        }
        
        # Si tiene datos de sentimiento
        if 'sentimiento_predicho' in df.columns:
            df_processed = preprocess_for_ml(df)
            
            stats.update({
                'processed_records': len(df_processed),
                'sentiment_distribution': df_processed['sentimiento_predicho'].value_counts().to_dict(),
                'avg_text_length': float(df_processed['transcription'].str.len().mean()),
                'min_text_length': int(df_processed['transcription'].str.len().min()),
                'max_text_length': int(df_processed['transcription'].str.len().max()),
                'ready_for_ml': len(df_processed) >= 10
            })
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# Agregar este endpoint adicional al final de app.py

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Limpiar cache de sentimientos"""
    global sentiment_cache
    with cache_lock:
        cache_size = len(sentiment_cache)
        sentiment_cache.clear()
    
    return jsonify({
        'success': True,
        'message': f'Cache limpiado. Se eliminaron {cache_size} entradas.'
    }), 200

@app.route('/export_model_report', methods=['GET'])
def export_model_report():
    """Exportar reporte del modelo en formato JSON"""
    try:
        if not os.path.exists('model_results.json'):
            return jsonify({'error': 'No hay resultados de modelo disponibles'}), 404
        
        with open('model_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Agregar timestamp
        results['export_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Información general del modelo y sistema"""
    return jsonify({
        'model_type': 'Logistic Regression with TF-IDF',
        'validation_method': 'Hold-out Split',
        'sentiment_analyzer': 'VADER + Machine Learning',
        'features': {
            'tfidf_max_features': 1000,
            'ngram_range': '(1, 2)',
            'preprocessing': 'Text cleaning, stopword removal'
        },
        'server_info': {
            'cache_size': len(sentiment_cache),
            'files_available': {
                'transcriptions': os.path.exists(TRANSCRIPTIONS_CSV),
                'model_results': os.path.exists('model_results.json')
            }
        }
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Iniciando servidor en puerto {port}")
    print("📊 Usando VADER Sentiment Analysis para mejor rendimiento")
    app.run(host='0.0.0.0', port=port, debug=False)