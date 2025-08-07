from flask import Flask, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_dance.contrib.google import make_google_blueprint, google
import os
import random
import string
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
from werkzeug.middleware.proxy_fix import ProxyFix
from oauthlib.oauth2.rfc6749.errors import TokenExpiredError

# Descargar recursos
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
spacy_es = spacy.load('es_core_news_sm')

# Configuración segura
if os.environ.get("FLASK_ENV") == "development":
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
else:
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# Inicializar app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_CLIENT_KEY", str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app, 
    supports_credentials=True,
    origins=["https://analisis-web.vercel.app"],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# Google OAuth
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
    redirect_to='google_login',
    scope=["profile", "email"]
)
app.register_blueprint(google_bp, url_prefix="/login/google/authorized")

# Rutas base
BASE_DIR = 'usuarios'
os.makedirs(BASE_DIR, exist_ok=True)


@app.route("/")
def index():
    return "API funcionando con login de Google"

@app.route("/me")
def me():
    if "user_email" not in session:
        return jsonify({"error": "No autenticado"}), 401
    return jsonify({
        "email": session["user_email"],
        "name": session.get("name"),
        "picture": session.get("picture")
    })

@app.route("/login/google/authorized")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    
    try:
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return jsonify({"error": "Failed to fetch user info"}), 400
            
        user_info = resp.json()
        email = user_info.get("email")
        
        if not email:
            return jsonify({"error": "Email not provided by Google"}), 400

        # Configura la sesión correctamente
        session['user_email'] = email
        session['name'] = user_info.get("name")
        session['picture'] = user_info.get("picture")
        session.permanent = True  # Hace la sesión persistente
        
        # Crea directorio de usuario si no existe
        user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
        os.makedirs(user_folder, exist_ok=True)
        
        return redirect(f"https://analisis-web.vercel.app/dashboard.html?email={email}")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_state():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))
 

@app.route('/login')
def login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    try:
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return redirect(url_for("google.login"))
    except TokenExpiredError:
        return redirect(url_for("google.login"))
    except Exception as e:
        return f"Error inesperado: {str(e)}", 500
    user_info = resp.json()
    email = user_info['email']
    session['user_email'] = email
    session['name'] = user_info.get('name')
    session['picture'] = user_info.get('picture')
    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    os.makedirs(user_folder, exist_ok=True)
    return redirect(f"https://analisis-web.vercel.app/dashboard.html?email={email}")

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({"message": "Logout successful"}), 200
   
def generate_state():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

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

@app.route('/create_project', methods=['POST'])
def create_project():
    if "user_email" not in session:
        return jsonify({"error": "No autenticado"}), 401
    
    data = request.get_json()
    project_name = data.get('nombre')
    
    if not project_name:
        return jsonify({"error": "Nombre de proyecto requerido"}), 400
    
    email = session['user_email']
    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    project_folder = os.path.join(user_folder, project_name)
    
    try:
        os.makedirs(project_folder, exist_ok=False)  # exist_ok=False para evitar sobrescritura
        return jsonify({"message": f"Proyecto '{project_name}' creado"}), 201
    except FileExistsError:
        return jsonify({"error": "El proyecto ya existe"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Obtener lista de proyectos del usuario
@app.route('/proyectos', methods=['GET'])
def get_proyectos():
    email = session.get('user_email')
    if not email:
        return jsonify({'error': 'Usuario no autenticado'}), 401

    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    
    if not os.path.isdir(user_folder):
        return jsonify({'projects': []})
    
    projects = []
    for name in os.listdir(user_folder):
        full_path = os.path.join(user_folder, name)
        if os.path.isdir(full_path):
            # Obtener información adicional del proyecto
            num_files = len(os.listdir(full_path)) if os.path.exists(full_path) else 0
            projects.append({
                'name': name,
                'file_count': num_files,
                'created_at': os.path.getctime(full_path)
            })
    
    return jsonify({'projects': projects})


# Obtener transcripciones de un proyecto específico
@app.route('/proyecto/<nombre>/transcripciones', methods=['GET'])
def get_transcripciones(nombre):
    email = session.get('user_email')
    if not email:
        return jsonify({'error': 'Usuario no autenticado'}), 401

    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    project_folder = os.path.join(user_folder, nombre)
    csv_path = os.path.join(project_folder, 'transcriptions.csv')

    if not os.path.isfile(csv_path):
        return jsonify({'error': 'No se encontró el archivo de transcripciones'}), 404

    df = pd.read_csv(csv_path)
    return jsonify(df.to_dict(orient='records'))


def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language='es-ES')
        except:
            return "[No se pudo transcribir]"

def save_transcription_to_csv(folder, filename, transcription):
    csv_path = os.path.join(folder, 'transcriptions.csv')
    df = pd.DataFrame([[filename, transcription]], columns=['filename', 'transcription'])
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

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

@app.route('/proyecto/<nombre>', methods=['DELETE'])
def delete_project(nombre):
    email = session.get('user_email')
    if not email:
        return jsonify({'error': 'Usuario no autenticado'}), 401

    user_folder = os.path.join(BASE_DIR, email.replace('@', '_at_'))
    project_folder = os.path.join(user_folder, nombre)
    
    try:
        if not os.path.exists(project_folder):
            return jsonify({'error': 'Proyecto no encontrado'}), 404
            
        shutil.rmtree(project_folder)
        return jsonify({'message': f'Proyecto {nombre} eliminado'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
