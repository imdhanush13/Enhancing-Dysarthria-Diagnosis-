import gradio as gr
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# Load trained model and scaler
model_1 = load_model("model.h5")
sc = joblib.load("sc.pkl")

def feature_extraction_mfcc(x):
    try:
        x, sr = librosa.load(x)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128)
        mfcc = np.mean(mfcc.T, axis=0)
    except:
        return None
    return mfcc

def predict_audio(file_path):
    x = feature_extraction_mfcc(file_path)
    if x is None or len(x) != 128:
        return "❌ Invalid audio or MFCC extraction failed"
    x_scaled = sc.transform([x])
    x_input = x_scaled.reshape(1, 16, 8, 1)
    pred = model_1.predict(x_input)[0][0]
    label = "Dysarthria" if pred > 0.5 else "Non-Dysarthria"
    return f"✅ Prediction: {label} (Confidence: {pred:.2f})"

# Gradio interface
interface = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload WAV Audio"),
    outputs="text",
    title="Dysarthria Detection from Audio",
    description="Upload a WAV file. Model uses MFCC + CNN.",
)

interface.launch(debug=True)

