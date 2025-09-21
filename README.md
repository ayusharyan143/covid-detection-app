# COVID-19 Chest X-Ray Detection Web App

A powerful deep learning web application built with Django and TensorFlow to classify chest X-ray images for the detection of COVID-19. This project compares multiple CNN architectures to provide fast and accurate predictions through a user-friendly interface.

**Developed by Ayush Aryan**

---

## 🚀 Live Demo

You can test the live application deployed on Render:

**[https://covid-detection-app.onrender.com/](https://covid-detection-app.onrender.com/)**

*(Note: Free-tier deployments may take a moment to spin up if inactive.)*

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Models & Performance](#models--performance)
* [Technology Stack](#technology-stack)
* [Screenshots](#screenshots)
* [Local Setup](#local-setup)
* [Usage](#usage)
* [Dataset Details](#dataset-details)
* [License](#license)
* [Contact](#contact)

---

## 📝 Project Overview

This project is a full-stack web application designed to aid in the preliminary screening of COVID-19 using chest X-ray images. It leverages the power of deep learning, specifically Convolutional Neural Networks (CNNs), to distinguish between X-rays of COVID-19 positive patients and healthy individuals. The application allows users to upload an image and select from four different pre-trained models, receiving a prediction, confidence score, and inference time in seconds.

---

## ✨ Key Features

- **Multi-Model Selection**: Choose from four distinct, trained deep learning models (Custom CNN, VGG16, Xception, ResNet50).
- **High Accuracy**: The top models achieve over **95% accuracy** on the test dataset.
- **Real-Time Predictions**: Get instant classification results upon uploading an image.
- **Detailed Analysis**: Results include the prediction, a confidence score, the model used, and inference time.
- **Fully Deployed**: The application is live and accessible online via Render.
- **User-Friendly Interface**: A clean and intuitive UI for easy image uploads and result viewing.

---

## 🧠 Models & Performance

Four different models were trained and evaluated on the same dataset to compare their performance. The VGG16 architecture and the Custom CNN provided the best results.

| Model             | Test Accuracy |
| ----------------- | :-----------: |
| 🥇 **VGG16** |   **95.19%** |
| 🥈 **Custom CNN** |   **95.05%** |
| 🥉 **Xception** |   **90.50%** |
| **ResNet50** |   **73.81%** |

---

## 🛠️ Technology Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)

---

## 📸 Screenshots

#### Main Interface & Upload
<img width="1283" height="548" alt="Main UI of the application" src="https://github.com/user-attachments/assets/885b1cda-d6f7-4ef5-9d13-8575a27b5ecb" />

#### Models (Custom CNN, Xception, ResNet50, and VGG16)
<img width="713" height="270" alt="Models (Custom CNN, Xception, ResNet50, and VGG16)" src="https://github.com/user-attachments/assets/14f99642-f6f4-4635-8276-5cb9758faac1" />

#### Prediction for a COVID-19 Positive Case
<img width="1249" height="874" alt="Prediction result for a COVID-19 case" src="https://github.com/user-attachments/assets/aca04c7f-23e7-4aa7-b507-60847a995575" />

#### Prediction for a Normal (Non-COVID) Case
<img width="1209" height="854" alt="Prediction result for a Normal case" src="https://github.com/user-attachments/assets/f944b254-2adf-46da-9741-1d4d7cd8a19f" />

---

## ⚙️ Local Setup

To run this project on your local machine, follow these steps.

### Prerequisites
* Python 3.8+
* Git

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ayusharyan143/covid-detection-app.git](https://github.com/ayusharyan143/covid-detection-app.git)
    cd covid-detection-app
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    venv\Scripts\activate
    # Or (Mac/Linux)
    # source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `train_models.py` script is included but not required to run the app, as the pre-trained models are already in the repository.*

4.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```
5.  **Start the development server:**
    ```bash
    python manage.py runserver
    ```
6.  Open your web browser and navigate to `http://127.0.0.1:8000/`.

---

## 🖥️ Usage

1.  Navigate to the web application (either the live demo or your local version).
2.  From the dropdown menu, select which AI model you'd like to use for the analysis.
3.  Click "Choose File" to upload a chest X-ray image from your computer.
4.  Click the "Analyze Image" button.
5.  The results, including the prediction, confidence score, model used, and inference time, will be displayed.

---

## 📁 Dataset Details

This project was trained on the **COVID-19 Radiography Database**, a large, publicly available collection of chest X-ray images. The dataset is a collaboration between researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh, along with medical doctors from Pakistan and Malaysia.

* **Source:** [Kaggle: COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **Total Images Used for Training:** 13,808 (3,616 COVID-19, 10,192 Normal)

### Citation
* M.E.H. Chowdhury, et al., "Can AI help in screening Viral and COVID-19 pneumonia?" IEEE Access, 2020.
* T. Rahman, et al., "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images," arXiv preprint, 2020.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 📬 Contact

Ayush Aryan - [GitHub Profile](https://github.com/ayusharyan143) - [LinkedIn Profile](https://www.linkedin.com/in/ayush-aryan-591344243/)