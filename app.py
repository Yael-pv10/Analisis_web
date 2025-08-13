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