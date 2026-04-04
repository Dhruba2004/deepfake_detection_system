import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import joblib
from pathlib import Path
import io
import soundfile as sf
import pickle

# --- NEW IMPORTS FOR TEXT DETECTION ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF warnings
try:
    import keras
    load_model = keras.models.load_model
    # Keras 3 moved pad_sequences to utils
    pad_sequences = keras.utils.pad_sequences 
except ImportError:
    st.error("Deep learning libraries not found. Run: pip install tensorflow keras")

# ==============================
# CONFIG
# ==============================

AUDIO_MODEL_DIR = Path("models/audio_models")
TEXT_MODEL_DIR = Path("models/text_models")
DEVICE = torch.device("cpu")

MAX_FRAMES = 150
SR = 16000

st.set_page_config(
    page_title="Deepfake Detection System",
    layout="centered"
)

# ==============================
# AUDIO MODELS DEFINITION
# ==============================

class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=20, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
    


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 8))  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(3072, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))



class OurMethodClassifier(nn.Module):
    def __init__(self, input_size=60, d_model=64, nhead=4, num_layers=2, num_classes=2, max_frames=150):
        super().__init__()
        self.max_frames = max_frames
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Embedding(max_frames, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes)
        )
    def forward(self, x):
        B, T, _ = x.shape
        if T > self.max_frames:
            x = x[:, :self.max_frames, :]
            T = self.max_frames
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.input_proj(x)
        x = x + self.pos_enc(positions)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ==============================
# AUDIO PREPROCESSING
# ==============================

def pad_or_truncate(feat, max_frames=150):
    if feat.shape[0] < max_frames:
        feat = np.pad(feat, ((0, max_frames - feat.shape[0]), (0, 0)))
    else:
        feat = feat[:max_frames]
    return feat

def load_audio(file):
    file.seek(0)
    audio, sr = sf.read(file)
    audio = np.array(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio

def preprocess_lstm(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20).T
    mfcc = pad_or_truncate(mfcc)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

def preprocess_transformer(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=60).T
    mfcc = pad_or_truncate(mfcc)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

def preprocess_cnn_audio(audio):
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = librosa.util.fix_length(spec, size=128, axis=1)
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)
    spec = torch.from_numpy(spec).float()
    spec = spec.unsqueeze(0).unsqueeze(0)
    return spec

# ==============================
# LOAD MODELS (AUDIO & TEXT)
# ==============================

@st.cache_resource
def load_audio_models():
    models = {}
    try:
        lstm = BiLSTMClassifier()
        lstm.load_state_dict(torch.load(AUDIO_MODEL_DIR / "lstm_model.pth", map_location="cpu"))
        lstm.eval()
        models["Bi-LSTM"] = lstm

        transformer = OurMethodClassifier()
        transformer.load_state_dict(torch.load(AUDIO_MODEL_DIR / "our_method_model.pth", map_location="cpu"))
        transformer.eval()
        models["Attention Model"] = transformer

        models["SVM"] = joblib.load(AUDIO_MODEL_DIR / "svm_model.pkl")
        models["Random Forest"] = joblib.load(AUDIO_MODEL_DIR / "rf_model.pkl")
        scaler = joblib.load(AUDIO_MODEL_DIR / "scaler.pkl")

        cnn_audio = CNNClassifier()
        cnn_audio.load_state_dict(torch.load(AUDIO_MODEL_DIR / "cnn_model.pth", map_location="cpu"), strict=False)
        cnn_audio.eval()
        models["Spectrogram-CNN"] = cnn_audio

        return models, scaler
    except Exception as e:
        return {}, None

@st.cache_resource
def load_text_models():
    models = {}
    
    try:
        # Vectorizers & Scalers
        tfidf_vectorizer = joblib.load(TEXT_MODEL_DIR / "tfidf_vectorizer.pkl")
        our_method_scaler = joblib.load(TEXT_MODEL_DIR / "our_method_scaler.pkl")
        with open(TEXT_MODEL_DIR / "lstm_tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Classical ML Models
        models["Naive Bayes"] = joblib.load(TEXT_MODEL_DIR / "naive_bayes.pkl")
        models["Logistic Regression"] = joblib.load(TEXT_MODEL_DIR / "logistic_regression.pkl")
        models["SVM"] = joblib.load(TEXT_MODEL_DIR / "svm_model.pkl")
        models["Our Method"] = joblib.load(TEXT_MODEL_DIR / "our_method_model.pkl")
        models["LSTM"] = keras.models.load_model(TEXT_MODEL_DIR / "lstm_model.keras", safe_mode=False)

        return models, tfidf_vectorizer, our_method_scaler, tokenizer
    except Exception as e:
        st.error(f"Error loading text models: {e}. Please ensure the files are correctly placed in {TEXT_MODEL_DIR}.")
        return {}, None, None, None

# ==============================
# PREDICTION FUNCTIONS
# ==============================

def predict_all_audio(models, scaler, audio):
    results = {}
    
    lstm_input = preprocess_lstm(audio)
    transformer_input = preprocess_transformer(audio)
    cnn_input = preprocess_cnn_audio(audio)

    results["Bi-LSTM"] = F.softmax(models["Bi-LSTM"](lstm_input), dim=1).detach().numpy()[0]
    results["Spectrogram-CNN"] = F.softmax(models["Spectrogram-CNN"](cnn_input), dim=1).detach().numpy()[0]
    results["Attention Model"] = F.softmax(models["Attention Model"](transformer_input), dim=1).detach().numpy()[0]

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        feat = np.mean(mfcc, axis=1).reshape(1, -1)
        feat_scaled = scaler.transform(feat)

        results["SVM"] = models["SVM"].predict_proba(feat_scaled)[0]
        results["Random Forest"] = models["Random Forest"].predict_proba(feat_scaled)[0]
    except Exception as e:
        results["SVM"] = np.array([0.5, 0.5])
        results["Random Forest"] = np.array([0.5, 0.5])

    return results

def predict_all_text(models, tfidf, scaler, tokenizer, text):
    results = {}
    
    # Text -> TF-IDF
    text_tfidf = tfidf.transform([text])
    
    # Naive Bayes
    results["Naive Bayes"] = models["Naive Bayes"].predict_proba(text_tfidf)[0]
    
    # Logistic Regression
    results["Logistic Regression"] = models["Logistic Regression"].predict_proba(text_tfidf)[0]
    
    # SVM (LinearSVC typically uses decision_function, fallback for binary)
    try:
        pred_svm = models["SVM"].predict(text_tfidf)[0]
        # Simulate probabilities based on prediction (assuming 0=Real, 1=Fake)
        results["SVM"] = np.array([1.0, 0.0]) if pred_svm == 0 else np.array([0.0, 1.0])
    except Exception:
        results["SVM"] = np.array([0.5, 0.5])

    # Our Method
    try:
        text_scaled = scaler.transform(text_tfidf.toarray())
        results["Our Method"] = models["Our Method"].predict_proba(text_scaled)[0]
    except Exception:
        results["Our Method"] = np.array([0.5, 0.5])

    # LSTM Preprocessing & Prediction
    try:
        seq = tokenizer.texts_to_sequences([text])
        # Using 300 based on standard shape inputs detected for this type
        padded_seq = pad_sequences(seq, maxlen=300) 
        lstm_pred = models["LSTM"].predict(padded_seq, verbose=0)[0]
        
        # Handle binary (1 output node) vs categorical (2 output nodes)
        if len(lstm_pred) == 1:
            prob_fake = lstm_pred[0]
            results["LSTM"] = np.array([1.0 - prob_fake, prob_fake])
        else:
            results["LSTM"] = lstm_pred
    except Exception as e:
        results["LSTM"] = np.array([0.5, 0.5])

    return results

# ==============================
# UI WITH TABS
# ==============================

st.title("Deepfake Detection System")
tabs = st.tabs(["🎵 Audio Detection", "🖼️ Image Detection", "📝 Text Detection"])

# ==============================
# AUDIO TAB
# ==============================
with tabs[0]:
    st.header("Fake Audio Detection")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"], key="audio")

    if uploaded_file is not None:
        audio = load_audio(uploaded_file)
        st.audio(uploaded_file)

        models, scaler = load_audio_models()
        if models:
            results = predict_all_audio(models, scaler, audio)

            st.subheader("DETECTION RESULTS")
            st.markdown("="*60)
            st.write(f"**File:** {uploaded_file.name}")
            st.markdown("="*60)

            votes = {"BONAFIDE": 0, "SPOOF": 0}

            for i, (name, prob) in enumerate(results.items(), start=1):
                bonafide = prob[0] * 100
                spoof = prob[1] * 100
                verdict = "BONAFIDE" if bonafide > spoof else "SPOOF"
                votes[verdict] += 1
                color = "🟢" if verdict == "BONAFIDE" else "🔴"

                st.markdown(f"""
                **METHOD {i} — {name}** Verdict : {color} {verdict}  
                Bonafide : {bonafide:.1f}% {'█' * int(bonafide // 5)}  
                Spoof    : {spoof:.1f}% {'█' * int(spoof // 5)}  
                """)

            st.markdown("="*60)

            total_models = len(results)
            bonafide_votes = votes["BONAFIDE"]
            spoof_votes = votes["SPOOF"]

            final_verdict = "BONAFIDE" if bonafide_votes > spoof_votes else "SPOOF"
            final_color = "🟢" if final_verdict == "BONAFIDE" else "🔴"

            st.markdown(f"""
            **MAJORITY VOTE ({total_models} models)** Bonafide votes : {bonafide_votes}/{total_models}  
            Spoof votes    : {spoof_votes}/{total_models}  

            ### ✔ FINAL VERDICT : {final_color} {final_verdict}
            """)
            st.markdown("="*60)
        else:
            st.warning("Audio models not found. Please check paths.")

# ==============================
# IMAGE TAB
# ==============================
with tabs[1]:
    st.header("Image Deepfake Detection")
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image")

    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        st.info("⚠️ Image detection model not added yet.")

# ==============================
# TEXT TAB (UPDATED)
# ==============================
with tabs[2]:
    st.header("Fake Text Detection")
    text_input = st.text_area("Enter text to analyze", height=200)

    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text using multi-model voting..."):
                t_models, tfidf, t_scaler, tokenizer = load_text_models()
                
                if t_models:
                    results = predict_all_text(t_models, tfidf, t_scaler, tokenizer, text_input)

                    st.subheader("DETECTION RESULTS")
                    st.markdown("="*60)
                    
                    votes = {"HUMAN-WRITTEN (REAL)": 0, "AI-GENERATED (FAKE)": 0}

                    for i, (name, prob) in enumerate(results.items(), start=1):
                        # Standard convention: index 0 is Real/Bonafide, index 1 is Fake/Spoof
                        real_score = prob[0] * 100
                        fake_score = prob[1] * 100
                        
                        verdict = "HUMAN-WRITTEN (REAL)" if real_score > fake_score else "AI-GENERATED (FAKE)"
                        votes[verdict] += 1
                        color = "🟢" if verdict == "HUMAN-WRITTEN (REAL)" else "🔴"

                        st.markdown(f"""
                        **METHOD {i} — {name}** Verdict : {color} {verdict}  
                        Real : {real_score:.1f}% {'█' * int(real_score // 5)}  
                        Fake : {fake_score:.1f}% {'█' * int(fake_score // 5)}  
                        """)

                    st.markdown("="*60)
                    
                    total_models = len(results)
                    real_votes = votes["HUMAN-WRITTEN (REAL)"]
                    fake_votes = votes["AI-GENERATED (FAKE)"]

                    final_verdict = "HUMAN-WRITTEN (REAL)" if real_votes > fake_votes else "AI-GENERATED (FAKE)"
                    final_color = "🟢" if final_verdict == "HUMAN-WRITTEN (REAL)" else "🔴"

                    st.markdown(f"""
                    **MAJORITY VOTE ({total_models} models)** Real votes : {real_votes}/{total_models}  
                    Fake votes : {fake_votes}/{total_models}  

                    ### ✔ FINAL VERDICT : {final_color} {final_verdict}
                    """)
                    st.markdown("="*60)