
Digit_Recognition-System

This is a simple deep learning-based micro project called Digit Recognition System. 
We were given a task to build a small project using Deep Learning techniques. 
For this, I chose to create a system that can recognize handwritten digits using a Convolutional Neural Network (CNN).

The project is trained using the MNIST dataset, which contains thousands of handwritten digit images from 0 to 9. 
The model learns how each digit looks and then can predict the digit from any new image given by the user.

The trained model is integrated into a web application built using Flask. 
Users can draw a digit on a canvas and submit it — the model will process the image and predict which digit was drawn. 
There's also a command-line script to test the model with external image files.

To run this project, you need to install the required Python libraries using the requirement.txt file. 
You can then either train the model from scratch using train.py, or use the already trained model (digit_model.h5). 
After that, start the web app using app.py.

This project helped me understand how to:
- Build and train CNN models
- Preprocess image data
- Deploy models using a web interface for real-time predictions

Technologies Used – Project Requirements:
- Python
- TensorFlow
- Flask
- NumPy
- Pillow
- MNIST Dataset

How to Run:

1. Install dependencies:
   pip install -r requirement.txt

2. Train the model (optional):
   python train.py

3. Start the Flask web application:
   python app.py

4. Open your browser and go to:
   http://127.0.0.1:5000

You will see a canvas where you can draw a number (0 to 9). 
After drawing, click submit, and it will show the predicted digit.

Where Can This Be Used?
- Educational tools for learning digit recognition and AI
- Digit-based form processing (e.g., postal codes, exam sheets)
- Mobile or tablet apps for math-based handwriting inputs
- ATM and check scanners to read handwritten numbers 
