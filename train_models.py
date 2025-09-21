# train_models.py
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# --- 1. CONFIGURATION ---
DATASET_DIR = r"C:\Users\ayush\Desktop\COVID_DETECTION_PROJECT\Dataset\COVID-19_Radiography_Dataset"
PROCESSED_DATA_DIR = "processed_data"
SAVE_MODEL_DIR = "predictor/model_weights"

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)


# --- 2. DATA PROCESSING MODULE ---
def process_and_save_data():
    print("Starting data processing...")
    def _load_images(path, urls, target):
        images, labels = [], []
        for url in urls:
            img = cv2.imread(os.path.join(path, url))
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                images.append(img)
                labels.append(target)
        return np.array(images), np.array(labels)

    covid_path = os.path.join(DATASET_DIR, "COVID", "images")
    normal_path = os.path.join(DATASET_DIR, "Normal", "images")
    covid_images, covid_labels = _load_images(covid_path, os.listdir(covid_path), 1)
    normal_images, normal_labels = _load_images(normal_path, os.listdir(normal_path), 0)
    
    data = np.concatenate([covid_images, normal_images], axis=0)
    labels = np.concatenate([covid_labels, normal_labels], axis=0)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42, stratify=labels)
    
    print("Saving preprocessed data to disk...")
    np.save(os.path.join(PROCESSED_DATA_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    print("✅ Data processing complete and files saved.")

def load_processed_data():
    print("Loading preprocessed data from disk...")
    x_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'x_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    x_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'x_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    print("✅ Data loaded successfully.")
    return x_train, x_test, y_train, y_test


# --- 3. MODEL CREATION MODULE ---
def create_custom_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Custom CNN model created.")
    return model

def create_vgg16_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ VGG16 model created.")
    return model

def create_resnet50_model():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ ResNet50 model created.")
    return model

def create_xception_model():
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Xception model created.")
    return model


# --- 4. TRAINING & EVALUATION MODULE ---
def train_and_evaluate_model(model, model_name, x_train, y_train, x_test, y_test):
    print(f"\n{'='*50}\n🔥 Starting Training for: {model_name.upper()}\n{'='*50}")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    model_path = os.path.join(SAVE_MODEL_DIR, f"{model_name}.keras")
    model.save(model_path)
    print(f"✅ Saved '{model_name}' model to {model_path}")
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🧪 Evaluation for {model_name}: Test Accuracy: {accuracy*100:.2f}%")
    
    y_pred_prob = model.predict(x_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'COVID'], yticklabels=['Normal', 'COVID'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

# --- 5. MAIN CONTROLLER ---
if __name__ == "__main__":
    # STEP 1: SKIPPED - This is already done.
    # process_and_save_data()
    
    # STEP 2: Load the data from disk.
    x_train, x_test, y_train, y_test = load_processed_data()

    # STEP 3: Train ONLY the new models.
    
    # --- Custom CNN and VGG16 are SKIPPED (already trained) ---
    
    # --- Train ResNet50 Model ---
    print("\n--- Preparing ResNet50 Model ---")
    resnet_model = create_resnet50_model()
    train_and_evaluate_model(resnet_model, "resnet50", x_train, y_train, x_test, y_test)
    
    # --- Train Xception Model ---
    print("\n--- Preparing Xception Model ---")
    xception_model = create_xception_model()
    train_and_evaluate_model(xception_model, "xception", x_train, y_train, x_test, y_test)

    print("\n🎉 All new training sessions complete.")