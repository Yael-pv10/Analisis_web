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
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

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
    """Preprocesar datos para machine learning con mejor filtrado"""
    try:
        # Filtrar solo las filas que tienen sentimiento procesado
        df_processed = df[
            (df['sentimiento_predicho'].notna()) & 
            (df['sentimiento_predicho'] != 'Sin procesar') &
            (df['sentimiento_predicho'] != '') &
            (df['sentimiento_predicho'] != 'Sin texto') &
            (df['transcription'].notna()) &
            (df['transcription'] != '') &
            (df['transcription'].str.len() > 0)  # Asegurar que no esté vacío
        ].copy()
        
        print(f"🔍 Registros después del filtro inicial: {len(df_processed)}")
        
        if len(df_processed) == 0:
            return df_processed
        
        # Limpiar textos
        df_processed['text_clean'] = df_processed['transcription'].apply(preprocess_text_fast)
        
        # Filtrar textos vacíos después del preprocesamiento
        df_processed = df_processed[
            (df_processed['text_clean'].notna()) &
            (df_processed['text_clean'].str.len() > 5) &  # Al menos 5 caracteres
            (df_processed['text_clean'].str.strip() != '')
        ]
        
        print(f"🔍 Registros después de limpiar texto: {len(df_processed)}")
        
        # Verificar distribución de clases
        if len(df_processed) > 0:
            class_counts = df_processed['sentimiento_predicho'].value_counts()
            print(f"📊 Distribución de clases: {dict(class_counts)}")
            
            # Filtrar clases con muy pocas muestras (menos de 2)
            valid_classes = class_counts[class_counts >= 2].index
            df_processed = df_processed[df_processed['sentimiento_predicho'].isin(valid_classes)]
            
            print(f"🔍 Registros después de filtrar clases pequeñas: {len(df_processed)}")
        
        return df_processed
        
    except Exception as e:
        print(f"❌ Error en preprocess_for_ml: {e}")
        return pd.DataFrame()  # Retornar DataFrame vacío en caso de error

def create_confusion_matrix_plot(y_true, y_pred, labels):
    """Crear matriz de confusión como imagen base64 con manejo robusto de errores"""
    try:
        plt.figure(figsize=(8, 6))
        
        # Crear matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Verificar si la matriz está vacía
        if cm.size == 0:
            plt.text(0.5, 0.5, 'No hay datos suficientes\npara la matriz de confusión', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=14, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
            plt.axis('off')
        else:
            # Crear heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Número de muestras'})
            
            plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Valores Reales', fontsize=12)
            plt.xlabel('Predicciones', fontsize=12)
            
            # Rotar etiquetas si son muy largas
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                   dpi=150, facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()  # Importante: cerrar la figura para liberar memoria
        
        return img_str
        
    except Exception as e:
        print(f"❌ Error creando matriz de confusión: {e}")
        
        # Crear imagen de error
        try:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, f'Error al crear\nmatriz de confusión:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=12, bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.7))
            plt.axis('off')
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                       dpi=150, facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except:
            return None

@app.route('/train_model', methods=['POST'])
def train_model():
    """Entrenar modelo de clasificación con Hold-out"""
    try:
        print("🚀 Iniciando entrenamiento del modelo...")
        
        # Verificar que existe el archivo
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            print("❌ Archivo transcriptions.csv no encontrado")
            return jsonify({'error': 'Archivo transcriptions.csv no encontrado'}), 400
        
        # Cargar datos
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        print(f"📊 Datos cargados: {len(df)} registros")
        
        # Verificar que hay datos con sentimientos
        if 'sentimiento_predicho' not in df.columns:
            print("❌ No hay datos de sentimiento")
            return jsonify({'error': 'No hay datos de sentimiento. Ejecuta el análisis primero.'}), 400
        
        # Preprocesar datos
        df_processed = preprocess_for_ml(df)
        print(f"🔧 Datos procesados: {len(df_processed)} registros válidos")
        
        if len(df_processed) < 10:
            print(f"❌ Datos insuficientes: {len(df_processed)} registros")
            return jsonify({'error': f'Datos insuficientes para entrenamiento. Solo {len(df_processed)} registros válidos.'}), 400
        
        # Preparar características y etiquetas
        X = df_processed['text_clean'].values
        y = df_processed['sentimiento_predicho'].values
        
        print(f"📝 Clases únicas: {np.unique(y)}")
        
        # Filtrar textos muy cortos que podrían causar problemas
        valid_indices = [i for i, text in enumerate(X) if len(text.strip()) > 3]
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"🔍 Después de filtrar textos cortos: {len(X)} registros")
        
        if len(X) < 10:
            return jsonify({'error': 'Muy pocos textos válidos después del filtrado'}), 400
        
        # Verificar que hay suficientes muestras por clase
        class_counts = pd.Series(y).value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            return jsonify({'error': f'Clase con muy pocas muestras: {class_counts.to_dict()}'}), 400
        
        # Vectorización TF-IDF con parámetros más conservadores
        vectorizer = TfidfVectorizer(
            max_features=min(500, len(X) // 2),  # Reducir características
            stop_words=list(stop_words_spanish),
            ngram_range=(1, 2),
            min_df=max(1, len(X) // 50),  # Ajustar min_df dinámicamente
            max_df=0.95,  # Ignorar términos muy frecuentes
            token_pattern=r'\b[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]{2,}\b'  # Solo palabras válidas
        )
        
        try:
            X_tfidf = vectorizer.fit_transform(X)
            print(f"🔤 Características TF-IDF: {X_tfidf.shape}")
        except Exception as e:
            print(f"❌ Error en vectorización: {e}")
            return jsonify({'error': f'Error en la vectorización TF-IDF: {str(e)}'}), 400
        
        # Verificar que la matriz no esté vacía
        if X_tfidf.nnz == 0:
            return jsonify({'error': 'La matriz TF-IDF resultó vacía. Revisa el preprocesamiento del texto.'}), 400
        
        # Codificar etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # División Hold-out con verificación
        test_size = float(request.json.get('test_size', 0.2))
        random_state = int(request.json.get('random_state', 42))
        
        # Asegurar que hay suficientes muestras para la división
        min_test_samples = max(1, min_class_count // 2)
        max_test_size = min(0.4, 1 - (min_test_samples / len(X)))
        test_size = min(test_size, max_test_size)
        
        print(f"📊 Test size ajustado: {test_size:.2f}")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y_encoded, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_encoded
            )
        except Exception as e:
            print(f"❌ Error en train_test_split: {e}")
            # Intentar sin stratify si falla
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y_encoded, 
                test_size=test_size, 
                random_state=random_state
            )
        
        print(f"📊 División: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Entrenar modelo de Regresión Logística con parámetros más robustos
        model = LogisticRegression(
            random_state=random_state,
            max_iter=2000,  # Más iteraciones
            class_weight='balanced',
            solver='liblinear',  # Solver más estable para datasets pequeños
            C=1.0  # Regularización moderada
        )
        
        try:
            model.fit(X_train, y_train)
            print("✅ Modelo entrenado")
        except Exception as e:
            print(f"❌ Error en entrenamiento del modelo: {e}")
            return jsonify({'error': f'Error al entrenar el modelo: {str(e)}'}), 500
        
        # Predicciones
        try:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        except Exception as e:
            print(f"❌ Error en predicciones: {e}")
            return jsonify({'error': f'Error en las predicciones: {str(e)}'}), 500
        
        # Métricas de desempeño
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"🎯 Precisión - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}")
        
        # Reporte de clasificación
        class_names = label_encoder.classes_
        try:
            classification_rep = classification_report(
                y_test, y_pred_test, 
                target_names=class_names,
                output_dict=True,
                zero_division=0  # Evitar errores de división por cero
            )
        except Exception as e:
            print(f"⚠️ Error en classification_report: {e}")
            classification_rep = {}
        
        # Matriz de confusión
        try:
            confusion_img = create_confusion_matrix_plot(
                label_encoder.inverse_transform(y_test),
                label_encoder.inverse_transform(y_pred_test),
                class_names
            )
        except Exception as e:
            print(f"⚠️ Error en matriz de confusión: {e}")
            confusion_img = None
        
        # Distribución de clases
        class_distribution = pd.Series(y).value_counts().to_dict()
        
        # Características más importantes (con manejo de errores)
        top_features = {}
        try:
            feature_names = vectorizer.get_feature_names_out()
            
            for i, class_name in enumerate(class_names):
                if len(class_names) == 2:
                    # Clasificación binaria
                    coefficients = model.coef_[0] if i == 1 else -model.coef_[0]
                else:
                    # Clasificación multiclase
                    coefficients = model.coef_[i]
                
                # Obtener índices de características más importantes
                top_indices = np.argsort(np.abs(coefficients))[-10:][::-1]
                
                top_features[class_name] = [
                    {
                        'feature': feature_names[idx],
                        'coefficient': float(coefficients[idx])
                    }
                    for idx in top_indices
                ]
        except Exception as e:
            print(f"⚠️ Error en características importantes: {e}")
            top_features = {}
        
        # Guardar resultados del modelo
        model_results = {
            'model_type': 'Logistic Regression',
            'train_size': int(X_train.shape[0]),
            'test_size': int(X_test.shape[0]),
            'total_features': int(X_tfidf.shape[1]),
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'class_distribution': class_distribution,
            'classification_report': classification_rep,
            'confusion_matrix_img': confusion_img,
            'top_features': top_features,
            'classes': list(class_names),
            'test_split_ratio': float(test_size),
            'random_state': random_state
        }
        
        # Guardar en archivo JSON para persistencia
        try:
            with open('model_results.json', 'w', encoding='utf-8') as f:
                # Crear una copia sin la imagen para el JSON
                json_results = model_results.copy()
                json_results.pop('confusion_matrix_img', None)
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            print("✅ Resultados guardados")
        except Exception as e:
            print(f"⚠️ Error al guardar resultados: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Modelo entrenado exitosamente',
            'results': model_results
        }), 200
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"❌ Error en get_model_results: {e}")
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
        print(f"❌ Error en predict_sentiment: {e}")
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
            # Filtrar datos procesados correctamente
            df_valid = df[
                (df['sentimiento_predicho'].notna()) & 
                (df['sentimiento_predicho'] != 'Sin procesar') &
                (df['sentimiento_predicho'] != '') &
                (df['sentimiento_predicho'] != 'Sin texto') &
                (df['transcription'].notna()) &
                (df['transcription'] != '')
            ]
            
            stats.update({
                'processed_records': len(df_valid),
                'sentiment_distribution': df_valid['sentimiento_predicho'].value_counts().to_dict(),
                'avg_text_length': float(df_valid['transcription'].str.len().mean()) if len(df_valid) > 0 else 0,
                'min_text_length': int(df_valid['transcription'].str.len().min()) if len(df_valid) > 0 else 0,
                'max_text_length': int(df_valid['transcription'].str.len().max()) if len(df_valid) > 0 else 0,
                'ready_for_ml': len(df_valid) >= 10
            })
        else:
            stats.update({
                'processed_records': 0,
                'ready_for_ml': False,
                'sentiment_distribution': {}
            })
        
        print(f"📊 Estadísticas generadas: {stats}")
        return jsonify(stats), 200
        
    except Exception as e:
        print(f"❌ Error en model_statistics: {e}")
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
    """Preprocesamiento rápido y más robusto"""
    try:
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ""
        
        # Convertir a string si no lo es
        text = str(text)
        
        # Convertir a minúsculas
        text = text.lower().strip()
        
        # Eliminar URLs, menciones, hashtags
        text = re.sub(r'http\S+|www\.\S+|@\w+|#\w+', '', text)
        
        # Eliminar números solos (pero mantener palabras con números)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Mantener solo letras, números y algunos signos de puntuación importantes
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\(\)áéíóúÁÉÍÓÚüÜñÑ]', ' ', text)
        
        # Limpiar espacios múltiples
        text = ' '.join(text.split())
        
        # Eliminar palabras muy cortas (menos de 2 caracteres)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        text = ' '.join(words)
        
        return text.strip()
        
    except Exception as e:
        print(f"⚠️ Error en preprocess_text_fast: {e}")
        return ""

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
        print(f"❌ Error en análisis de sentimiento: {e}")
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
        import traceback
        traceback.print_exc()
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
            distribucion = df['sentimiento_predicho'].value_counts().to_dict()
            info['distribucion_sentimientos'] = distribucion
            
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
            'has_sentiment_columns': all(col in df.columns for col in ['sentimiento_predicho', 'confianza', 'rank_sentimiento']),
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
            'has_sentiment_columns': all(col in df.columns for col in ['sentimiento_predicho', 'confianza', 'rank_sentimiento']),
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
        'cache_size': len(sentiment_cache),
        'transcriptions_file_exists': os.path.exists(TRANSCRIPTIONS_CSV),
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S')
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
    
    print(f"🧹 Cache limpiado: {cache_size} entradas eliminadas")
    
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
            },
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'dependencies': {
                'flask': True,
                'pandas': True,
                'sklearn': True,
                'vader_sentiment': True,
                'matplotlib': True,
                'seaborn': True
            }
        }
    }), 200

# Endpoint adicional para debugging y monitoreo
@app.route('/debug_info', methods=['GET'])
def debug_info():
    """Información de debugging del sistema"""
    try:
        info = {
            'server_status': 'running',
            'current_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'working_directory': os.getcwd(),
            'files_in_directory': os.listdir('.'),
            'cache_info': {
                'size': len(sentiment_cache),
                'sample_keys': list(sentiment_cache.keys())[:5] if sentiment_cache else []
            }
        }
        
        # Información del archivo de transcripciones
        if os.path.exists(TRANSCRIPTIONS_CSV):
            df = pd.read_csv(TRANSCRIPTIONS_CSV)
            info['transcriptions_info'] = {
                'exists': True,
                'rows': len(df),
                'columns': list(df.columns),
                'file_size_bytes': os.path.getsize(TRANSCRIPTIONS_CSV),
                'has_sentiment_data': 'sentimiento_predicho' in df.columns
            }
        else:
            info['transcriptions_info'] = {'exists': False}
        
        # Información del modelo
        if os.path.exists('model_results.json'):
            info['model_info'] = {
                'results_file_exists': True,
                'file_size_bytes': os.path.getsize('model_results.json')
            }
        else:
            info['model_info'] = {'results_file_exists': False}
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'debug': True}), 500

# Endpoint para resetear todo el sistema (útil para testing)
@app.route('/reset_system', methods=['POST'])
def reset_system():
    """Resetear cache y archivos temporales (solo para desarrollo)"""
    try:
        # Limpiar cache
        global sentiment_cache
        with cache_lock:
            sentiment_cache.clear()
        
        # Eliminar archivos temporales (opcional)
        temp_files = ['model_results.json']
        removed_files = []
        
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                removed_files.append(file)
        
        return jsonify({
            'success': True,
            'message': 'Sistema reseteado',
            'cache_cleared': True,
            'files_removed': removed_files
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Importar sys para información del sistema
import sys

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🚀 Iniciando servidor en puerto {port}")
    print(f"📊 Usando VADER Sentiment Analysis para mejor rendimiento")
    print(f"🔧 Modo debug: {debug_mode}")
    print(f"📁 Directorio de trabajo: {os.getcwd()}")
    print(f"📝 Archivo de transcripciones: {TRANSCRIPTIONS_CSV}")
    
    # Verificar dependencias importantes
    try:
        import sklearn
        print(f"✅ scikit-learn versión: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn no está instalado")
    
    try:
        import matplotlib
        print(f"✅ matplotlib versión: {matplotlib.__version__}")
    except ImportError:
        print("❌ matplotlib no está instalado")
    
    try:
        import seaborn
        print(f"✅ seaborn versión: {seaborn.__version__}")
    except ImportError:
        print("❌ seaborn no está instalado")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)