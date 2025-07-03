import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import soundfile as sf
import tempfile

model = tf.keras.models.load_model("model.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def extract_features(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1, 1)

st.title("🗣️ Spoken Digit Recognition (Audio Classifier)")
st.write("Upload a .wav file of a spoken digit (0–9) to classify.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    features = extract_features(tmp_path)
    prediction = model.predict(features)
    predicted_digit = le.inverse_transform([np.argmax(prediction)])

    st.success(f"Predicted Digit: **{predicted_digit[0]}**")
