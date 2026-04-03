import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import joblib
from pathlib import Path

# ==============================
# CONFIG
# ==============================

MODEL_DIR = Path("models")
DEVICE = torch.device("cpu")

MAX_FRAMES = 150
SR = 16000

st.set_page_config(
    page_title="Fake Audio Detection",
    layout="centered"
)

# ==============================
# LSTM MODEL
# ==============================

class BiLSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
    
    # ==============================
# CNN AUDIO MODEL (Spectrogram)
# ==============================

# ==============================
# CNN AUDIO MODEL (Spectrogram)
# ==============================

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 🔥 FIX: 6 * 8 = 48. Flattened: 64 channels * 48 = 3072
            nn.AdaptiveAvgPool2d((6, 8))  
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 🔥 FIX: Accepts the 3072 features expected by the checkpoint
            nn.Linear(3072, 128),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
# ==============================
# OUR METHOD (TRANSFORMER)
# ==============================

class OurMethodClassifier(nn.Module):

    def __init__(
        self,
        input_size=60,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_frames=150
    ):

        super().__init__()

        self.max_frames = max_frames

        self.input_proj = nn.Linear(
            input_size,
            d_model
        )

        self.pos_enc = nn.Embedding(
            max_frames,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        B, T, _ = x.shape

        if T > self.max_frames:
            x = x[:, :self.max_frames, :]
            T = self.max_frames

        positions = torch.arange(
            T,
            device=x.device
        ).unsqueeze(0).expand(B, T)

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
        feat = np.pad(
            feat,
            ((0, max_frames - feat.shape[0]), (0, 0))
        )
    else:
        feat = feat[:max_frames]

    return feat



import io
import soundfile as sf

def load_audio(file):

    file.seek(0)

    audio, sr = sf.read(file)

    # Ensure numpy array
    audio = np.array(audio)

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Resample
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    return audio



def preprocess_lstm(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=20
    ).T

    mfcc = pad_or_truncate(mfcc)

    return torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)




def preprocess_transformer(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=60
    ).T

    mfcc = pad_or_truncate(mfcc)

    return torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)

def preprocess_cnn_audio(audio):


    # Ensure numpy float32
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # Mel Spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=128
    )

    spec = librosa.power_to_db(spec, ref=np.max)

    spec = librosa.util.fix_length(spec, size=128, axis=1)

    # Normalize
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)

    # 🔥 CONVERT TO TORCH TENSOR
    spec = torch.from_numpy(spec).float()

    # Add channel + batch dimension
    spec = spec.unsqueeze(0).unsqueeze(0)

    return spec


# ==============================
# LOAD MODELS
# ==============================


@st.cache_resource

def load_models():

    models = {}

    # LSTM
    lstm = BiLSTMClassifier()
    lstm.load_state_dict(
        torch.load(
            MODEL_DIR / "lstm_model.pth",
            map_location="cpu"
        )
    )
    lstm.eval()
    models["Bi-LSTM"] = lstm

    # Transformer
    transformer = OurMethodClassifier()
    transformer.load_state_dict(
        torch.load(
            MODEL_DIR / "our_method_model.pth",
            map_location="cpu"
        )
    )
    transformer.eval()
    models["Attention Model"] = transformer

    # Classical models
    models["SVM"] = joblib.load(
        MODEL_DIR / "svm_model.pkl"
    )

    models["Random Forest"] = joblib.load(
        MODEL_DIR / "rf_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    # CNN AUDIO
    cnn_audio = CNNClassifier()
   # Temporary fix
    cnn_audio.load_state_dict(
    torch.load(MODEL_DIR / "cnn_model.pth", map_location="cpu"),
    strict=False
)
    cnn_audio.eval()

    models["Spectrogram-CNN"] = cnn_audio


    return models, scaler


# ==============================
# PREDICTION
# ==============================


def predict_all(models, scaler, audio):
    results = {}

    # --- 1. DEEP LEARNING MODELS (No Scaler Needed) ---
    lstm_input = preprocess_lstm(audio)
    transformer_input = preprocess_transformer(audio)
    cnn_input = preprocess_cnn_audio(audio)

    results["Bi-LSTM"] = F.softmax(models["Bi-LSTM"](lstm_input), dim=1).detach().numpy()[0]
    results["Spectrogram-CNN"] = F.softmax(models["Spectrogram-CNN"](cnn_input), dim=1).detach().numpy()[0]
    results["Attention Model"] = F.softmax(models["Attention Model"](transformer_input), dim=1).detach().numpy()[0]

    # --- 2. CLASSICAL MODELS (SCALER INTEGRATION) ---
    try:
        # Extract features - IMPORTANT: n_mfcc must match your training (usually 20 or 40)
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        feat = np.mean(mfcc, axis=1).reshape(1, -1)

        # Apply the scaler from your models folder
        feat_scaled = scaler.transform(feat)

        # Get probabilities from SVM and RF using the scaled features
        results["SVM"] = models["SVM"].predict_proba(feat_scaled)[0]
        results["Random Forest"] = models["Random Forest"].predict_proba(feat_scaled)[0]
        
    except Exception as e:
        st.error(f"Error in Scaler/Classical models: {e}")
        # Fallback so the app doesn't crash
        results["SVM"] = np.array([0.5, 0.5])
        results["Random Forest"] = np.array([0.5, 0.5])

    return results


# ==============================
# UI WITH TABS
# ==============================

st.title("Deepfake Detection System")

tabs = st.tabs(["🎵 Audio Detection", "🖼️ Image Detection", "📝 Text Detection"])


# ==============================
# AUDIO TAB (YOUR ORIGINAL CODE)
# ==============================

with tabs[0]:

    st.header("Fake Audio Detection")

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "flac"],
        key="audio"
    )

    if uploaded_file is not None:

        audio = load_audio(uploaded_file)
        st.audio(uploaded_file)

        models, scaler = load_models()
        results = predict_all(models, scaler, audio)

        st.subheader("DETECTION RESULTS")
        st.markdown("="*60)

        file_name = uploaded_file.name
        st.write(f"**File:** {file_name}")

        st.markdown("="*60)

        votes = {"BONAFIDE": 0, "SPOOF": 0}

        for i, (name, prob) in enumerate(results.items(), start=1):

            bonafide = prob[0] * 100
            spoof = prob[1] * 100

            verdict = "BONAFIDE" if bonafide > spoof else "SPOOF"
            votes[verdict] += 1

            color = "🟢" if verdict == "BONAFIDE" else "🔴"

            st.markdown(f"""
**METHOD {i} — {name}**  
Verdict : {color} {verdict}  
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
**MAJORITY VOTE ({total_models} models)**  
Bonafide votes : {bonafide_votes}/{total_models}  
Spoof votes    : {spoof_votes}/{total_models}  

### ✔ FINAL VERDICT : {final_color} {final_verdict}
""")

        st.markdown("="*60)


# ==============================
# IMAGE TAB (UI ONLY)
# ==============================

with tabs[1]:

    st.header("Image Deepfake Detection")

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="image"
    )

    if image_file is not None:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        st.info("⚠️ Image detection model not added yet.")


# ==============================
# TEXT TAB (UI ONLY)
# ==============================

with tabs[2]:

    st.header("Fake Text Detection")

    text_input = st.text_area(
        "Enter text to analyze",
        height=200
    )

    if text_input:
        st.info("⚠️ Text detection model not added yet.")