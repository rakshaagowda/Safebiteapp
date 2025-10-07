#just integrated with model
'''

import os
import numpy as np
from keras.models import load_model
#from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_model_and_labels(model_path: str):
    """
    Load Keras model and optional labels list.
    Returns (labels_list, model) or (None, None) if model missing.
    """
    if not model_path or not os.path.exists(model_path):
        print("Model not found:", model_path)
        return None, None

    model = load_model(model_path)

    labels = None
    labels_path = os.path.join(os.path.dirname(model_path), "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    return labels, model


def predict_food_from_path(image_path: str, model, labels=None, target_size=(224, 224)):
    """
    Predict the food class from an image file path using the trained model.
    Returns (food_name, confidence).
    """
    if model is None:
        return "unknown_food", 0.0

    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    if labels and idx < len(labels):
        return labels[idx], conf

    return f"class_{idx}", conf
'''

#after most integration

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# -------------------- Paths -------------------- #
MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_classifier.h5")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "labels.txt")


def load_model_and_labels(model_path: str = MODEL_PATH):
    """
    Load a trained Keras model and optional labels list.
    Returns (labels_list, model) or (None, None) if model missing.
    """
    if not model_path or not os.path.exists(model_path):
        print("Model not found:", model_path)
        return None, None

    # Load Keras model
    model = load_model(model_path)

    # Load labels from 'labels.txt' if exists
    labels = None
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    return labels, model


def predict_food_from_path(image_path: str, model, labels=None, target_size=(224, 224)):
    """
    Predict the food class from an image file path using the trained model.
    Returns (food_name, confidence).
    """
    if model is None:
        return "unknown_food", 0.0

    # Load and preprocess image
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    # Map index to label if labels available
    if labels and idx < len(labels):
        return labels[idx], conf

    return f"class_{idx}", conf


# -------------------- Example Usage -------------------- #
if __name__ == "__main__":
    labels, model = load_model_and_labels(MODEL_PATH)

    test_image = "test_food.jpg"  # Replace with your test image path
    if os.path.exists(test_image):
        food, confidence = predict_food_from_path(test_image, model, labels)
        print(f"Predicted: {food} (Confidence: {confidence:.2f})")
    else:
        print("Test image not found:", test_image)
