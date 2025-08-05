from flask import Flask, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_dance.contrib.google import make_google_blueprint, google
import os
import uuid
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Descarga de recursos necesarios
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
spacy_es = spacy.load('es_core_news_sm')

# Configuración base
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get('FLASK_CLIENT_KEY')

# Configurar OAuth Google
blueprint = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
    redirect_url="https://audio-preprocessing-api-production.up.railway.app/login/google/authorized",
    scope=["profile", "email"]
)
app.register_blueprint(blueprint, url_prefix="/login")

# Carpeta base para usuarios
BASE_DIR = 'usuarios'
os.makedirs(BASE_DIR, exist_ok=True)

# Ruta para login con Google
@app.route('/login')
def login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return "Error de autenticación", 403
    user_info = resp.json()
    email = user_info['email']
    session['user_email'] = email
    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    os.makedirs(user_folder, exist_ok=True)
    return redirect(f"https://analisis-web.vercel.app/dashboard.html?email={email}")

# Ruta para logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect("https://analisis-web.vercel.app/index.html")

# Ruta para subir audios
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files or 'project' not in request.form:
        return jsonify({'error': 'Missing file or project name'}), 400

    email = session.get('user_email')
    if not email:
        return jsonify({'error': 'Unauthorized'}), 401

    file = request.files['file']
    project = request.form['project']
    original_ext = file.filename.split('.')[-1].lower()
    unique_name = f"{uuid.uuid4()}.{original_ext}"

    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    project_folder = os.path.join(user_folder, project)
    os.makedirs(project_folder, exist_ok=True)
    audio_path = os.path.join(project_folder, unique_name)
    file.save(audio_path)

    if original_ext in ['webm', 'mp3']:
        audio_wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        try:
            sound = AudioSegment.from_file(audio_path)
            sound.export(audio_wav_path, format='wav')
            audio_path = audio_wav_path
        except Exception as e:
            return jsonify({'error': f'Error al convertir audio: {e}'}), 500

    transcription = transcribe_audio(audio_path)
    save_transcription_to_csv(project_folder, file.filename, transcription)
    return jsonify({'transcription': transcription}), 200

# Transcripción
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language='es-ES')
        except:
            return "[No se pudo transcribir]"

# Guardar transcripción en CSV
def save_transcription_to_csv(folder, filename, transcription):
    csv_path = os.path.join(folder, 'transcriptions.csv')
    df = pd.DataFrame([[filename, transcription]], columns=['filename', 'transcription'])
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

# Preprocesamiento completo paso a paso
@app.route('/preprocesamiento', methods=['GET'])
def preprocesamiento():
    email = session.get('user_email')
    project = request.args.get('project')
    if not email or not project:
        return jsonify({'error': 'Missing user session or project'}), 400

    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    project_folder = os.path.join(user_folder, project)
    csv_path = os.path.join(project_folder, 'transcriptions.csv')

    if not os.path.isfile(csv_path):
        return jsonify({'error': 'No transcriptions found'}), 404

    df = pd.read_csv(csv_path)
    df['lowercase'] = df['transcription'].str.lower()
    df['tokens'] = df['lowercase'].apply(word_tokenize)
    stop_words = set(stopwords.words('spanish'))
    df['no_stopwords'] = df['tokens'].apply(lambda tokens: [w for w in tokens if w not in stop_words and w not in string.punctuation])
    df['lemmas'] = df['no_stopwords'].apply(lambda tokens: [spacy_es(word)[0].lemma_ for word in tokens])
    df['final'] = df['lemmas'].apply(lambda tokens: ' '.join(tokens))

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['final'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    result = {
        'original': df[['filename', 'transcription']].to_dict(orient='records'),
        'lowercase': df['lowercase'].tolist(),
        'tokens': df['tokens'].tolist(),
        'no_stopwords': df['no_stopwords'].tolist(),
        'lemmas': df['lemmas'].tolist(),
        'tfidf': tfidf_df.to_dict(orient='records')
    }
    return jsonify(result)

# Ejecutar servidor
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
