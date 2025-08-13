from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import pandas as pd
import os
from flask_cors import CORS
from pydub import AudioSegment
import uuid
from ftfy import fix_text
from transformers import pipeline
import threading
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar el modelo una sola vez al inicio
logger.info("Cargando modelo BERT...")
try:
    classifier = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1  # Forzar CPU para evitar problemas de GPU
    )
    logger.info("Modelo BERT cargado exitosamente")
except Exception as e:
    logger.error(f"Error cargando BERT: {e}")
    classifier = None

# Preprocesamiento
import nltk
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Inicializaciones
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nlp = spacy.load("es_core_news_sm")
    stop_words = set(stopwords.words('spanish'))
except Exception as e:
    logger.error(f"Error inicializando NLTK/spaCy: {e}")

# Configuración del servidor
app = Flask(__name__)
CORS(app)

# Variables globales
tasks = {}  # Para tracking de tareas asíncronas

AUDIO_DIR = 'audios'
TRANSCRIPTIONS_CSV = 'transcriptions.csv'
PROCESSED_CSV = 'processed_transcriptions.csv'
SENTIMENTS_CSV = 'opiniones_con_sentimientos.csv'
os.makedirs(AUDIO_DIR, exist_ok=True)

traduccion_sentimientos = {
    "1 star": "Muy negativo",
    "2 stars": "Negativo", 
    "3 stars": "Neutral",
    "4 stars": "Positivo",
    "5 stars": "Muy positivo"
}

# Mapeo de ranking
rank_map = {
    "1 star": 1,
    "2 stars": 2,
    "3 stars": 3,
    "4 stars": 4,
    "5 stars": 5
}

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

def predecir_sentimiento_batch(textos, batch_size=10):
    """Procesa sentimientos en lotes para mejorar eficiencia"""
    if not classifier:
        return [{"label": "Error: Modelo no disponible", "rank": None, "confidence": 0} for _ in textos]
    
    resultados = []
    total_textos = len(textos)
    
    logger.info(f"Procesando {total_textos} textos en lotes de {batch_size}")
    
    for i in range(0, total_textos, batch_size):
        batch = textos[i:i+batch_size]
        batch_limpio = []
        indices_validos = []
        
        # Filtrar textos válidos
        for j, texto in enumerate(batch):
            if pd.isna(texto) or str(texto).strip() == "":
                resultados.append({"label": "No disponible", "rank": None, "confidence": 0})
            else:
                batch_limpio.append(str(texto))
                indices_validos.append(len(resultados))
                resultados.append(None)  # Placeholder
        
        # Procesar lote válido
        if batch_limpio:
            try:
                logger.info(f"Procesando lote {i//batch_size + 1} de {(total_textos-1)//batch_size + 1}")
                predicciones = classifier(batch_limpio)
                
                for idx, prediccion in zip(indices_validos, predicciones):
                    etiqueta = prediccion["label"]
                    confidence = prediccion["score"]
                    resultados[idx] = {
                        "label": traduccion_sentimientos.get(etiqueta, etiqueta),
                        "rank": rank_map.get(etiqueta, 0),
                        "confidence": confidence
                    }
                    
            except Exception as e:
                logger.error(f"Error procesando lote: {e}")
                for idx in indices_validos:
                    resultados[idx] = {"label": "Error", "rank": None, "confidence": 0}
        
        # Pequeña pausa para evitar sobrecarga
        time.sleep(0.1)
    
    return resultados

def predecir_sentimiento(texto):
    """Función individual mantenida para compatibilidad"""
    if pd.isna(texto) or str(texto).strip() == "":
        return {"label": "No disponible", "rank": None, "confidence": 0}
    
    if not classifier:
        return {"label": "Error: Modelo no disponible", "rank": None, "confidence": 0}
    
    try:
        resultado = classifier(str(texto))[0]
        etiqueta = resultado["label"]
        confidence = resultado["score"]
        return {
            "label": traduccion_sentimientos.get(etiqueta, etiqueta),
            "rank": rank_map.get(etiqueta, 0),
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error con texto: {str(texto)[:30]}... -> {e}")
        return {"label": "Error", "rank": None, "confidence": 0}

def process_sentiment_task(task_id):
    """Ejecuta el análisis y guarda el resultado en tasks"""
    try:
        file_path = TRANSCRIPTIONS_CSV
        if not os.path.exists(file_path):
            tasks[task_id] = {"status": "error", "error": "Archivo CSV no encontrado"}
            return

        df = pd.read_csv(file_path)
        if "transcription" not in df.columns:
            tasks[task_id] = {"status": "error", "error": "Columna 'transcription' no encontrada"}
            return

        logger.info(f"Iniciando análisis para {len(df)} transcripciones")
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 0

        # Limpiar texto
        df["transcription"] = df["transcription"].apply(
            lambda x: fix_text(str(x)) if pd.notna(x) else x
        )

        # Usar procesamiento en lotes
        textos = df["transcription"].tolist()
        resultados = predecir_sentimiento_batch(textos, batch_size=8)  # Lotes más pequeños
        
        # Asignar resultados
        df["sentimiento_predicho"] = [r["label"] for r in resultados]
        df["rank"] = [r["rank"] for r in resultados]
        df["confidence"] = [r["confidence"] for r in resultados]

        # Guardar archivo
        df.to_csv(SENTIMENTS_CSV, index=False, encoding='utf-8-sig')
        logger.info("Análisis completado y archivo guardado")

        tasks[task_id] = {"status": "completed", "result": df.to_dict(orient='records'), "progress": 100}

    except Exception as e:
        logger.error(f"Error en process_sentiment_task: {e}")
        tasks[task_id] = {"status": "error", "error": str(e)}

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "progress": 0}

    # Lanza análisis en segundo plano
    threading.Thread(target=process_sentiment_task, args=(task_id,)).start()

    return jsonify({"task_id": task_id}), 200

@app.route('/get_analysis/<task_id>', methods=['GET'])
def get_analysis(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Tarea no encontrada"}), 404
    return jsonify(task), 200

# OPTIMIZADO: Endpoint para analizar sentimientos con procesamiento en lotes
@app.route('/analyze_sentiments', methods=['POST'])
def analyze_sentiments():
    try:
        if not os.path.exists(TRANSCRIPTIONS_CSV):
            return jsonify({"error": "No hay transcripciones disponibles"}), 404
        
        df = pd.read_csv(TRANSCRIPTIONS_CSV)
        if "transcription" not in df.columns:
            return jsonify({"error": "Columna 'transcription' no encontrada"}), 400
        
        logger.info(f"Analizando {len(df)} transcripciones")
        
        # Limpiar texto
        df["transcription"] = df["transcription"].apply(
            lambda x: fix_text(str(x)) if pd.notna(x) else x
        )
        
        # Usar procesamiento en lotes optimizado
        textos = df["transcription"].tolist()
        resultados = predecir_sentimiento_batch(textos, batch_size=8)
        
        df["sentimiento_predicho"] = [r["label"] for r in resultados]
        df["rank"] = [r["rank"] for r in resultados]
        df["confidence"] = [r["confidence"] for r in resultados]
        
        # Guardar archivo
        df.to_csv(SENTIMENTS_CSV, index=False, encoding='utf-8-sig')
        
        logger.info("Análisis completado")
        return jsonify(df.to_dict(orient='records')), 200
        
    except Exception as e:
        logger.error(f"Error en analyze_sentiments: {e}")
        return jsonify({"error": str(e)}), 500

# NUEVO: Endpoint para obtener progreso en tiempo real
@app.route('/get_analysis_progress/<task_id>', methods=['GET'])
def get_analysis_progress(task_id):
    task = tasks.get(task_id, {"status": "not_found"})
    return jsonify(task)

# NUEVO: Endpoint para obtener datos ya analizados
@app.route('/get_sentiment_data', methods=['GET'])
def get_sentiment_data():
    try:
        if os.path.exists(SENTIMENTS_CSV):
            df = pd.read_csv(SENTIMENTS_CSV)
            return jsonify(df.to_dict(orient='records')), 200
        else:
            return jsonify([]), 200
    except Exception as e:
        logger.error(f"Error obteniendo datos de sentimiento: {e}")
        return jsonify({"error": str(e)}), 500

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

@app.route('/preprocessing_steps', methods=['GET'])
def preprocessing_steps():
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'No hay archivo de transcripciones'}), 404

    df = pd.read_csv(TRANSCRIPTIONS_CSV)
    raw_texts = df['transcription'].tolist()
    results = []
    processed_texts = []

    for text in raw_texts:
        step = {'original': text}

        # Minúsculas
        text_lower = text.lower()
        step['lower'] = text_lower

        # Tokenización
        tokens = word_tokenize(text_lower)
        step['tokens'] = tokens

        # Eliminar signos de puntuación y stopwords
        tokens_clean = [t for t in tokens if t.isalpha() and t not in stop_words]
        step['no_stopwords'] = tokens_clean

        # Lematización con spaCy
        doc = nlp(' '.join(tokens_clean))
        lemmatized = [token.lemma_ for token in doc]
        step['lemmatized'] = lemmatized

        processed_texts.append(' '.join(lemmatized))
        results.append(step)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Guardar CSV final
    df['processed'] = processed_texts
    df.to_csv(PROCESSED_CSV, index=False)

    return jsonify({
        'steps': results,
        'tfidf': tfidf_df.to_dict(orient='records')
    })

@app.route('/download_processed_csv', methods=['GET'])
def download_processed_csv():
    if os.path.exists(PROCESSED_CSV):
        return send_file(PROCESSED_CSV, as_attachment=True)
    return jsonify({'error': 'Archivo procesado no encontrado'}), 404

@app.route('/download_sentiments_csv', methods=['GET'])
def download_sentiments_csv():
    if os.path.exists(SENTIMENTS_CSV):
        return send_file(SENTIMENTS_CSV, as_attachment=True)
    return jsonify({'error': 'Archivo de sentimientos no encontrado'}), 404

# NUEVO: Endpoint para verificar el estado del modelo
@app.route('/model_status', methods=['GET'])
def model_status():
    return jsonify({
        "model_loaded": classifier is not None,
        "model_name": "nlptown/bert-base-multilingual-uncased-sentiment" if classifier else None
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)