import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_digit(image_path):
    # Load model
    model = load_model('models/model.h5')
    
    # Process image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # Invert colors
    img = img.reshape(1, 28, 28, 1) / 255.0
    
    # Predict
    pred = model.predict(img)
    return np.argmax(pred)

if __name__ == '__main__':
    image_path = input("Enter image path: ")
    print(f"Predicted digit: {predict_digit(image_path)}")