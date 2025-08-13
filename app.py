from flask import Flask, request, jsonify, send_file
import speech_recognition as sr
import pandas as pd
import os
from flask_cors import CORS
from pydub import AudioSegment
import uuid
from ftfy import fix_text
import re
import threading
import time

# Importar modelo más rápido
try:
    # Usar TextBlob como alternativa rápida
    from textblob import TextBlob
    USE_TEXTBLOB = True
    print("✅ Usando TextBlob para análisis rápido")
except ImportError:
    # Fallback a transformers si TextBlob no está disponible
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    USE_TEXTBLOB = False
    print("⚠️ Usando transformers (más lento)")

# Preprocesamiento optimizado
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
except:
    pass

try:
    nlp = spacy.load("es_core_news_sm")
except:
    # Si no está instalado spacy en español, usar uno básico
    nlp = None

stop_words = set(stopwords.words('spanish')) if nltk.data.find('corpora/stopwords') else set()

# Configuración del servidor
app = Flask(__name__)
CORS(app)

AUDIO_DIR = 'audios'
TRANSCRIPTIONS_CSV = 'transcriptions.csv'
PROCESSED_CSV = 'processed_transcriptions.csv'
os.makedirs(AUDIO_DIR, exist_ok=True)

# Estado global para procesamiento asíncrono
processing_status = {
    'is_processing': False,
    'progress': 0,
    'total': 0,
    'results': None
}

def predecir_sentimiento_rapido(texto):
    """Análisis de sentimientos optimizado y rápido"""
    if pd.isna(texto) or str(texto).strip() == "":
        return {"label": "Neutral", "rank": 2, "score": 0.5}
    
    try:
        texto = str(texto).lower().strip()
        
        if USE_TEXTBLOB:
            # TextBlob es mucho más rápido
            blob = TextBlob(texto)
            polarity = blob.sentiment.polarity
            
            # Convertir polarity (-1 a 1) a categorías
            if polarity <= -0.3:
                return {"label": "Negativo", "rank": 1, "score": abs(polarity)}
            elif polarity >= 0.3:
                return {"label": "Positivo", "rank": 3, "score": polarity}
            else:
                return {"label": "Neutral", "rank": 2, "score": abs(polarity)}
        
        else:
            # Análisis básico con palabras clave (muy rápido)
            palabras_positivas = {
                'excelente', 'bueno', 'genial', 'fantástico', 'increíble', 
                'perfecto', 'maravilloso', 'espectacular', 'satisfecho', 
                'contento', 'feliz', 'amor', 'encanta', 'recomiendo'
            }
            
            palabras_negativas = {
                'malo', 'terrible', 'horrible', 'pésimo', 'odio', 
                'detesto', 'awful', 'disgusto', 'molesto', 'enojado', 
                'triste', 'decepcionado', 'insatisfecho', 'problema'
            }
            
            # Contar palabras positivas y negativas
            palabras = set(re.findall(r'\b\w+\b', texto.lower()))
            pos_count = len(palabras.intersection(palabras_positivas))
            neg_count = len(palabras.intersection(palabras_negativas))
            
            if neg_count > pos_count:
                return {"label": "Negativo", "rank": 1, "score": neg_count/(pos_count+neg_count+1)}
            elif pos_count > neg_count:
                return {"label": "Positivo", "rank": 3, "score": pos_count/(pos_count+neg_count+1)}
            else:
                return {"label": "Neutral", "rank": 2, "score": 0.5}
                
    except Exception as e:
        print(f"Error con texto: {texto[:30]}... -> {e}")
        return {"label": "Neutral", "rank": 2, "score": 0.5}

def procesar_sentimientos_lote(df, batch_size=50):
    """Procesar sentimientos en lotes para mejor rendimiento"""
    global processing_status
    
    processing_status['is_processing'] = True
    processing_status['progress'] = 0
    processing_status['total'] = len(df)
    
    resultados = []
    
    try:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Procesar lote
            for idx, row in batch.iterrows():
                resultado = predecir_sentimiento_rapido(row['transcription'])
                resultados.append(resultado)
                processing_status['progress'] += 1
            
            # Pequeña pausa para no sobrecargar
            time.sleep(0.01)
        
        # Aplicar resultados al DataFrame
        df["sentimiento_predicho"] = [r["label"] for r in resultados]
        df["rank"] = [r["rank"] for r in resultados]
        df["confidence"] = [r["score"] for r in resultados]
        
        # Guardar resultados
        output_path = "opiniones_con_sentimientos.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        processing_status['results'] = df.to_dict(orient='records')
        processing_status['is_processing'] = False
        
        print(f"✅ Procesamiento completado. {len(df)} registros procesados.")
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
        processing_status['is_processing'] = False
        processing_status['results'] = None

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

@app.route('/analyze_sentiments', methods=['POST'])
def analyze_sentiments():
    print("🔹 Iniciando análisis de sentimientos optimizado...")

    # Verificar si el archivo CSV existe
    file_path = "transcriptions.csv"
    if not os.path.exists(file_path):
        print(f"❌ No se encontró el archivo: {file_path}")
        return jsonify({'error': 'Archivo CSV no encontrado'}), 400

    try:
        print(f"📂 Leyendo archivo: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Archivo leído correctamente. Filas: {len(df)}")

        # Verificar columna transcription
        if "transcription" not in df.columns:
            print("❌ La columna 'transcription' no existe en el CSV")
            return jsonify({'error': "Columna 'transcription' no encontrada"}), 400

        # Limpiar datos vacíos
        df = df.dropna(subset=['transcription'])
        df = df[df['transcription'].str.strip() != '']
        
        print(f"📊 Registros válidos para procesar: {len(df)}")

        # Si son pocos registros, procesar síncronamente
        if len(df) <= 20:
            print("⚡ Procesamiento síncrono (pocos registros)")
            df["transcription"] = df["transcription"].apply(
                lambda x: fix_text(str(x)) if pd.notna(x) else x
            )
            
            resultados = df["transcription"].apply(predecir_sentimiento_rapido)
            df["sentimiento_predicho"] = resultados.apply(lambda x: x["label"])
            df["rank"] = resultados.apply(lambda x: x["rank"])
            df["confidence"] = resultados.apply(lambda x: x["score"])
            
            # Guardar CSV
            output_path = "opiniones_con_sentimientos.csv"
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            return jsonify(df.to_dict(orient='records')), 200
        
        else:
            # Procesar asíncronamente para muchos registros
            print("🔄 Iniciando procesamiento asíncrono...")
            df["transcription"] = df["transcription"].apply(
                lambda x: fix_text(str(x)) if pd.notna(x) else x
            )
            
            # Iniciar procesamiento en hilo separado
            thread = threading.Thread(
                target=procesar_sentimientos_lote, 
                args=(df,)
            )
            thread.start()
            
            return jsonify({
                'status': 'processing',
                'message': 'Procesamiento iniciado',
                'total_records': len(df)
            }), 202

    except Exception as e:
        print(f"❌ Error en analyze_sentiments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/processing_status', methods=['GET'])
def get_processing_status():
    """Endpoint para verificar el estado del procesamiento"""
    global processing_status
    
    if processing_status['results'] is not None and not processing_status['is_processing']:
        # Procesamiento completado
        results = processing_status['results']
        processing_status['results'] = None  # Limpiar después de enviar
        return jsonify({
            'status': 'completed',
            'data': results
        }), 200
    
    elif processing_status['is_processing']:
        # Procesamiento en curso
        progress_percent = (processing_status['progress'] / processing_status['total']) * 100
        return jsonify({
            'status': 'processing',
            'progress': processing_status['progress'],
            'total': processing_status['total'],
            'progress_percent': round(progress_percent, 2)
        }), 200
    
    else:
        return jsonify({
            'status': 'idle',
            'message': 'No hay procesamiento en curso'
        }), 200

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

        # Tokenización básica
        tokens = re.findall(r'\b\w+\b', text_lower)
        step['tokens'] = tokens

        # Eliminar stopwords si están disponibles
        if stop_words:
            tokens_clean = [t for t in tokens if t not in stop_words and len(t) > 2]
        else:
            tokens_clean = [t for t in tokens if len(t) > 2]
        step['no_stopwords'] = tokens_clean

        processed_texts.append(' '.join(tokens_clean))
        results.append(step)

    # TF-IDF optimizado
    try:
        vectorizer = TfidfVectorizer(max_features=100)  # Limitar features para rapidez
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    except:
        tfidf_df = pd.DataFrame()

    # Guardar CSV final
    df['processed'] = processed_texts
    df.to_csv(PROCESSED_CSV, index=False)

    return jsonify({
        'steps': results[:10],  # Limitar resultados para rapidez
        'tfidf': tfidf_df.head(10).to_dict(orient='records')
    })

@app.route('/download_processed_csv', methods=['GET'])
def download_processed_csv():
    if os.path.exists(PROCESSED_CSV):
        return send_file(PROCESSED_CSV, as_attachment=True)
    return jsonify({'error': 'Archivo procesado no encontrado'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)