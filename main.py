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



def load_audio(file):

    audio, _ = librosa.load(
        file,
        sr=SR
    )

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


    return models, scaler


# ==============================
# PREDICTION
# ==============================


def predict_all(models, scaler, audio):

    results = {}

    lstm_input = preprocess_lstm(audio)
    transformer_input = preprocess_transformer(audio)

    # LSTM
    prob = F.softmax(
        models["Bi-LSTM"](lstm_input),
        dim=1
    ).detach().numpy()[0]

    results["Bi-LSTM"] = prob

    # Transformer
    prob = F.softmax(
        models["Attention Model"](transformer_input),
        dim=1
    ).detach().numpy()[0]

    results["Attention Model"] = prob

    # Classical feature
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=40
    )

    feat = np.mean(
        mfcc,
        axis=1
    ).reshape(1, -1)

    feat = scaler.transform(feat)

    # SVM
    prob = models["SVM"].predict_proba(feat)[0]
    results["SVM"] = prob

    # Random Forest
    prob = models["Random Forest"].predict_proba(feat)[0]
    results["Random Forest"] = prob

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