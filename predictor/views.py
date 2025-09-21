# predictor/views.py
import os
import cv2
import time
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model # type: ignore

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    'custom_cnn': os.path.join(BASE_DIR, 'model_weights', 'custom_cnn.keras'),
    'vgg16': os.path.join(BASE_DIR, 'model_weights', 'vgg16.keras'),
    'resnet50': os.path.join(BASE_DIR, 'model_weights', 'resnet50.keras'), # Add this line
    'xception': os.path.join(BASE_DIR, 'model_weights', 'xception.keras'),   # Add this line
}
MODELS = {}

for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        try:
            MODELS[name] = load_model(path)
            print(f"✅ Model '{name}' loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model '{name}': {e}")
    else:
        print(f"⚠️ Warning: Model file not found for '{name}' at {path}")

# --- VIEW LOGIC ---
IMAGE_SIZE = 100
covid_pred_labels = ['Non Covid-19', 'Covid-19']

# predictor/views.py

def index(request):
    if request.method == 'POST' and request.FILES.get('ImgFile'):
        img_file = request.FILES['ImgFile']
        fs = FileSystemStorage()
        filename = fs.save(img_file.name, img_file)
        img_path_relative = fs.url(filename)
        img_path_absolute = os.path.join(settings.MEDIA_ROOT, filename)
        
        selected_model_name = request.POST.get('model_choice', 'custom_cnn')
        model = MODELS.get(selected_model_name)

        if not model:
            return render(request, 'index.html', {'error': f"Model '{selected_model_name}' is not available."})

        # Preprocess the image
        img = cv2.imread(img_path_absolute)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        
        # Make prediction
        start_time = time.time()
        prediction_score = model.predict(img_input)[0][0]
        end_time = time.time()
        exec_time = end_time - start_time

        # Interpret results
        if prediction_score > 0.5:
            predicted_class = covid_pred_labels[1]
            confidence_score = prediction_score * 100
        else:
            predicted_class = covid_pred_labels[0]
            confidence_score = (1 - prediction_score) * 100
        
        # We create a more user-friendly name to display
        model_display_name = selected_model_name.replace('_', ' ').title()
        
        context = {
            'v_pred': predicted_class,
            'v_cf': f"{confidence_score:.2f}",
            'v_time': f"{exec_time:.2f}",
            'image_url': img_path_relative,
            'v_model_name': model_display_name,  # <-- ADD THIS LINE
        }
        return render(request, 'index.html', context)
    
    return render(request, "index.html")