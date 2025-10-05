'''import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array


MODEL_PATH = os.path.join("backend", "models", "food_classifier.h5")
LABELS_PATH = os.path.join("backend", "models", "labels.txt")

_model = None
_labels = None

def _load_model_and_labels():
    global _model, _labels
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _labels is None:
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = [line.strip() for line in f.readlines() if line.strip()]
    return _model, _labels

def predict_food(image_path):
    model, labels = _load_model_and_labels()
    img = image.load_img(image_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    food_name = labels[idx] if idx < len(labels) else "unknown"
    return food_name, confidence
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore

# Paths to your model and labels
MODEL_PATH = os.path.join("backend", "models", "food_classifier.h5")
LABELS_PATH = os.path.join("backend", "models", "labels.txt")

# Global variables to store loaded model and labels
_model = None
_labels = None

def _load_model_and_labels():
    """
    Load the Keras model and labels from disk (only once).

    Raises:
        FileNotFoundError: If model or labels file is missing.

    Returns:
        tuple: (model, labels)
    """
    global _model, _labels

    # Load model if not already loaded
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)

    # Load labels if not already loaded
    if _labels is None:
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = [line.strip() for line in f.readlines() if line.strip()]

    return _model, _labels

def predict_food(image_path):
    """
    Predict the food in an image and return its name and confidence.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (food_name: str, confidence: float)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    model, labels = _load_model_and_labels()

    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    food_name = labels[idx] if idx < len(labels) else "unknown"

    return food_name, confidence

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     test_image = "test_image.jpg"
#     food, conf = predict_food(test_image)
#     print(f"Predicted food: {food}, Confidence: {conf:.2f}")
