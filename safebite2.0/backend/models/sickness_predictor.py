#Ml
'''
import os
import pickle

# Example of a simple heuristic-based predictor:
# - If any food known to be high-risk (spoiled_food / raw seafood / undercooked) appears often -> higher risk
# - If very high fat/sugar frequency -> slight increase
HIGH_RISK_ITEMS = {"spoiled_food", "raw_shellfish", "undercooked_meat", "expired_milk"}

def predict_sickness(logs, model=None):
    """
    logs: list of log entries
    model: optional trained ML model to use instead of heuristic
    returns: (risk_prob [0-1], risky_foods_list, recommendations_list)
    """
    if model is not None:
        # Expect model to take a feature vector computed from logs; for demo we assume model.predict_proba returns [p_not, p_yes]
        try:
            feats = _features_from_logs(logs)
            prob = model.predict_proba([feats])[0][1]
            flagged = []  # could be determined from model
            recs = _recommendations_from_flagged(flagged)
            return float(prob), flagged, recs
        except Exception as e:
            print("Sickness model failed:", e)

    # Heuristic:
    counts = {}
    for l in logs:
        f = l.get("food", "").lower()
        counts[f] = counts.get(f, 0) + 1
    risky = [f for f in counts if f in HIGH_RISK_ITEMS or counts[f] > 7]  # eaten >7 times in history
    risk = 0.75 if risky else 0.05
    recs = _recommendations_from_flagged(risky)
    return risk, risky, recs

def _recommendations_from_flagged(risky_foods):
    recs = []
    if not risky_foods:
        recs.append("Maintain balanced diet and monitor symptoms.")
        return recs
    for f in risky_foods:
        recs.append(f"Avoid {f}. If symptoms appear, hydrate, rest, and consult a doctor if severe.")
    recs.append("Increase fiber, drink water, and avoid heavy fried meals for 48 hours.")
    return recs

def _features_from_logs(logs):
    # simple synthetic feature creation (counts)
    from collections import Counter
    cnt = Counter(l.get("food","").lower() for l in logs)
    # produce a fixed-length vector by hashing top items (demo)
    items = sorted(cnt.items(), key=lambda x:-x[1])[:20]
    feats = [c for _, c in items]
    # pad to 20
    feats += [0]*(20-len(feats))
    return feats

def load_sickness_model_if_any(path):
    if path and os.path.exists(path):
        try:
            with open(path,"rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("Failed loading sickness model:", e)
    return None
'''

#DL CNN model
# train_sickness_predictor.py
import os
import warnings
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# -------------------- Suppress Warnings -------------------- #
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# -------------------- Paths -------------------- #
BASE_DIR = os.path.join("..", "food-101", "food-101")  # Adjusted to match your folder structure
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TRAIN_DIR = os.path.join(BASE_DIR, "train_sickness")
TEST_DIR = os.path.join(BASE_DIR, "test_sickness")
MODEL_PATH = os.path.join("..", "models", "sickness_cnn_model.h5")

# -------------------- Parameters -------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
SPLIT_RATIO = 0.8  # Train/Test split

# -------------------- Create train/test directories -------------------- #
if not os.path.exists(TRAIN_DIR):
    print("🟢 Creating train/test directories for sickness prediction...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Use only directories (ignore files like .DS_Store)
    categories = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]

    # For demo: only pick a few categories considered high-risk
    high_risk_foods = ["raw_shellfish", "spoiled_food", "burger", "pizza"]
    categories = [c for c in categories if c.lower() in high_risk_foods]

    for category in categories:
        os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

        imgs = os.listdir(os.path.join(IMAGES_DIR, category))
        random.shuffle(imgs)
        split_idx = int(SPLIT_RATIO * len(imgs))

        for img in imgs[:split_idx]:
            shutil.copy(os.path.join(IMAGES_DIR, category, img),
                        os.path.join(TRAIN_DIR, category, img))
        for img in imgs[split_idx:]:
            shutil.copy(os.path.join(IMAGES_DIR, category, img),
                        os.path.join(TEST_DIR, category, img))
    print("✅ Dataset split complete.")

# -------------------- Data Preprocessing -------------------- #
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -------------------- Model -------------------- #
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# -------------------- Compile -------------------- #
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- Callbacks -------------------- #
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -------------------- Train -------------------- #
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# -------------------- Cleanup -------------------- #
tf.keras.backend.clear_session()
print(f"✅ Training completed. Model saved as '{MODEL_PATH}'")
