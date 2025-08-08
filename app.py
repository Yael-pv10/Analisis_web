from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import pandas as pd
import os
from flask_cors import CORS
from pydub import AudioSegment
import uuid
from transformers import pipeline
from ftfy import fix_text

# Preprocesamiento
import nltk
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Inicializaciones
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load("es_core_news_sm")
stop_words = set(stopwords.words('spanish'))

# Configuración del servidor
app = Flask(__name__)
CORS(app)

AUDIO_DIR = 'audios'
TRANSCRIPTIONS_CSV = 'transcriptions.csv'
PROCESSED_CSV = 'processed_transcriptions.csv'
os.makedirs(AUDIO_DIR, exist_ok=True)

# Modelo de análisis de sentimientos
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

traduccion_sentimientos = {
    "1 star": "Muy negativo",
    "2 stars": "Negativo",
    "3 stars": "Neutral",
    "4 stars": "Positivo",
    "5 stars": "Muy positivo"
}
rank_map = {
    "1 star": 1,
    "2 stars": 2,
    "3 stars": 3,
    "4 stars": 4,
    "5 stars": 5
}

def predecir_sentimiento(texto):
    if pd.isna(texto) or texto.strip() == "":
        return {"label": "No disponible", "rank": None}
    try:
        resultado = classifier(texto)[0]["label"]
        return {
            "label": traduccion_sentimientos.get(resultado, resultado),
            "rank": rank_map.get(resultado)
        }
    except Exception as e:
        print(f"Error con texto: {texto[:30]}... -> {e}")
        return {"label": "Error", "rank": None}

@app.route('/ver_resultados_sentimientos', methods=['GET'])
def ver_resultados_sentimientos():
    if not os.path.exists(TRANSCRIPTIONS_CSV):
        return jsonify({'error': 'No hay archivo de transcripciones'}), 404

    df = pd.read_csv(TRANSCRIPTIONS_CSV)

    # Arreglar texto
    df["transcription"] = df["transcription"].apply(lambda x: fix_text(str(x)) if pd.notna(x) else x)

    # Aplicar modelo
    resultados = df["transcription"].apply(predecir_sentimiento)
    df["sentimiento_predicho"] = resultados.apply(lambda x: x["label"])
    df["rank"] = resultados.apply(lambda x: x["rank"])

    # Guardar CSV para descarga si se desea
    df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')

    # Mostrar resultados en una página HTML
    return render_template('Analizar_Sentimientos.html', tablas=[df.to_html(classes='table table-striped', index=False, justify='center')], titles=df.columns.values)

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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
