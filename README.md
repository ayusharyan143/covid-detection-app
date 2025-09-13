# COVID-19 Chest X-Ray Detection App

## Developed by Ayush Aryan

I am Ayush Aryan, the developer behind this project. This COVID-19 Chest X-Ray Detection App uses cutting-edge deep learning techniques to detect COVID-19 from chest X-ray images. My goal was to create a user-friendly and efficient application to help in the early detection of COVID-19 using Convolutional Neural Networks (CNN).


## Overview

This project uses deep learning to detect COVID-19 from chest X-ray images. By leveraging a Convolutional Neural Network (CNN) model trained on X-ray image datasets, this app classifies chest X-ray images into two categories:

- **COVID-19**
- **Normal (Non-COVID-19)**

The app allows users to upload a chest X-ray image, which is then processed by a pre-trained CNN model to make a prediction. The result is displayed alongside the confidence score and inference time.

<img width="1339" height="538" alt="image" src="https://github.com/user-attachments/assets/39d7cbae-0c03-413e-89ce-7103c2b20542" />



<img width="1608" height="754" alt="image" src="https://github.com/user-attachments/assets/fa7e26ce-3cc9-4058-b3b6-13eafbaba97e" />


## Dataset

### COVID-19 Radiography Database

The dataset used for this project is the **COVID-19 Radiography Database**, a publicly available collection of chest X-ray images for COVID-19, normal, and viral pneumonia cases. This dataset is continually updated and is a collaboration between several universities, medical institutions, and research teams across the globe.

- **COVID-19 Images**: 3,616 images
- **Normal Images**: 10,192 images
- **Viral Pneumonia Images**: 1,345 images
- **Lung Opacity Images** (Non-COVID lung infection): 6,012 images

The dataset is designed to assist researchers in developing AI models for detecting COVID-19 using X-ray imaging. It has been utilized in the creation of this app to build a robust model that can classify COVID-19 from other lung infections and healthy lungs.

For more information, or if you'd like to access the dataset, visit the following links:

- **COVID-19 Radiography Database**: [Download Dataset](https://doi.org/10.34740/kaggle/dsv/3122958)
- **COVID-QU-Ex Dataset**: [Kaggle Repository](https://doi.org/10.34740/kaggle/dsv/3122958)

### Dataset Citation

Please cite the following articles if you use this dataset:

1. M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. [Paper Link](https://ieeexplore.ieee.org/document/9216327)
2. Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. [arXiv Preprint Link](https://arxiv.org/abs/2012.02238)


## Features

- **User-friendly Interface**: Upload X-ray images and view predictions with just a few clicks.
- **Fast Prediction**: Get instant results with real-time model inference.
- **Confidence Scores**: View the confidence score of the model's prediction.
- **Detailed Results**: Inference time is also displayed for transparency.

## Technologies Used

- **Python**: The primary programming language for the backend logic.
- **TensorFlow/Keras**: For training the deep learning model and performing image predictions.
- **OpenCV**: Used for image preprocessing, such as resizing and normalization.
- **Django**: Framework for the web application and handling user uploads.
- **Matplotlib**: Used for displaying the image.
- **HTML/CSS**: For designing the frontend interface.
- **VSCode**: A popular code editor for writing and managing Python code.
- **PyCharm**: An integrated development environment (IDE) for Python development.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.x
- pip (Python package manager)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayusharyan143/covid-detection-app.git
   cd covid-detection-app

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the app:**
   ```bash
   python manage.py runserver
   
# How to Use
1. Go to the app's homepage in your browser.
2. Click on the "Choose File" button to upload a chest X-ray image.
3. After uploading the image, click the "Analyze Image" button.
4. The model will process the image, and the prediction (COVID or Normal) will be displayed along with the confidence score and inference time.

# Model Details
The model used in this project is a pre-trained Convolutional Neural Network (CNN). The architecture and weights are saved in two files:

- `model_architecture.json`: Contains the model architecture in JSON format.
- `model_weights.weights.h5`: Contains the weights of the trained model.

The model is loaded dynamically from these files when the app is run.

# Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository, create a new branch, and submit a pull request. Ensure that your code follows the existing style and passes the tests.

# Bug Reports
If you encounter any bugs or issues, feel free to open an issue on the repository's Issue Tracker.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- TensorFlow and Keras for providing the tools to build the deep learning model.
- OpenCV for image preprocessing.
- Django for creating the web framework.
- All contributors who helped improve this project.

# Snapshoot:

### COVID-19 : 
<img width="1449" height="699" alt="image" src="https://github.com/user-attachments/assets/ae917b8f-83c5-412e-8d57-ed575afeffdf" />

### NON COVID-19( NORMAL ) : 
<img width="1390" height="697" alt="image" src="https://github.com/user-attachments/assets/5ecc4f34-fe35-4248-93ef-6a082c832b7f" />





