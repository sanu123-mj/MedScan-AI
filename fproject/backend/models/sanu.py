app OverflowError           












from flask import Flask, request, jsonify
import tensorflow as tf
from backend.models.skincancer import create_model


  # Import your model function from skincancer.py
import numpy as np
import cv2
import os

app = Flask(__name__)

# Create and compile the model (you may want to load weights if needed)
model = create_model()

# Optionally, load model weights (if you've saved them separately)
model.load_weights('path_to_your_weights.h5')  # Use the correct path if you have saved weights

@app.route('/')
def home():
    return "Welcome to the Skin Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Update size based on your model input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = int(np.argmax(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(np.max(prediction))
    })

if __name__ == '__main__':
    app.run(debug=True)
