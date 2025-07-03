# 🗣️ Audio Digit Recognition Web App

A Streamlit app that classifies spoken digits (0–9) using a CNN trained on the Free Spoken Digit Dataset (FSDD).

## 🚀 Features
- Upload `.wav` files for prediction
- Uses MFCCs + CNN for classification
- Trained model accuracy: ~90%

## 📦 Requirements
- `model.h5`
- `label_encoder.pkl`

## 🛠️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy to Streamlit Cloud
1. Push files to GitHub
2. Visit https://streamlit.io/cloud
3. Deploy and get a public link
