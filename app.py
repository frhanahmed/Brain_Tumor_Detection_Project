import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
model = load_model('brain_tumor_model.h5')
img_size = 128  # same size used for training

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, (img_size, img_size)) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    label = "Tumor" if np.argmax(prediction) == 1 else "No Tumor"

    return label

if __name__ == '__main__':
    app.run(debug=True)