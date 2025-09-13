# type: ignore

import os
import cv2
import time
from tensorflow.keras.models import model_from_json
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Define class names
covid_pred = ['Covid-19', 'Non Covid-19']
IMAGE_SIZE = 100  # Image size for resizing

# Paths to model and weights files
cnn_model = r"C:\Users\ayush\Desktop\Detecting COVID-19 From Chest X-Ray\covidApp\predictor\model_weights\model_weights.weights.h5"
cnn_json = r"C:\Users\ayush\Desktop\Detecting COVID-19 From Chest X-Ray\covidApp\predictor\model_weights\model_architecture.json"

# Helper functions
def read_image(filepath):
    return cv2.imread(filepath)

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

def clear_mediadir():
    media_dir = './media'
    for f in os.listdir(media_dir):
        os.remove(os.path.join(media_dir, f))

# Define the main view
def index(request):
    if request.method == 'POST':
        # Clear previous media files
        clear_mediadir()

        # Retrieve the uploaded file
        img = request.FILES['ImgFile']
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        img_path = fs.path(filename)

        # Initialize the prediction array
        pred_arr = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        im = read_image(img_path)

        if im is not None:
            # Resize image to match model input size
            pred_arr[0] = resize_image(im, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize the image data
        pred_arr = pred_arr / 255.0

        # Load the model architecture and weights
        cnn_start = time.time()

        with open(cnn_json, 'r') as json_file:
            model = model_from_json(json_file.read())

        model.load_weights(cnn_model)
        label_cnn = model.predict(pred_arr)

        # Get the predicted class and confidence score
        idx_cnn = np.argmax(label_cnn[0])
        cf_score_cnn = np.amax(label_cnn[0])
        cnn_end = time.time()

        # Calculate inference time
        cnn_exec = cnn_end - cnn_start

        # Check confidence score and handle low-confidence predictions
        if cf_score_cnn < 0.6:
            predicted_class = "Non Covid-19"  # If the confidence is low
            cf_score_cnn = cf_score_cnn
        else:
            predicted_class = covid_pred[idx_cnn]
            cf_score_cnn = cf_score_cnn

        # Log the prediction results
        print(f"Prediction: {predicted_class}")
        print(f"Confidence Score: {cf_score_cnn:.8f}%")
        print(f"Prediction Time: {cnn_exec:.2f} sec")
        print(f"Image Path: {img_path}")

        # Prepare the response context
        response = {
            'table': "table",
            'col0': "Metric",
            'col1': "COVID Prediction",
            'col2': "Confidence Score",
            'col3': "Inference Time",
            'v_pred': predicted_class,
            'v_cf': round(cf_score_cnn, 8),
            'v_time': round(cnn_exec, 2),
            'image': "../media/" + img.name,
        }

        return render(request, 'index.html', response)

    else:
        return render(request, "index.html")
