from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'model/digit_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
model = load_model(model_path)

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # You must have templates/index.html

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to match MNIST input
        img_array = 255 - np.array(img)  # Invert colors (white background, black digits)
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Normalize

        # Predict using the model
        pred = model.predict(img_array)
        predicted_digit = int(np.argmax(pred))

        return jsonify({'digit': predicted_digit})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
