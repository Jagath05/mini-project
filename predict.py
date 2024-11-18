import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = r"D:\project\DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI-main\Code\model\vit_deepfake_model.h5"

# Load model
model = load_model(MODEL_PATH)

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Predict image
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    return "Fake" if prediction > 0.5 else "Real"

# Predict video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224)) / 255.0
        pred = model.predict(np.expand_dims(frame, axis=0))[0][0]
        predictions.append(pred)
    cap.release()
    avg_pred = np.mean(predictions)
    return "Fake" if avg_pred > 0.5 else "Real"
