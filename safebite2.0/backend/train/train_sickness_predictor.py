# train_sickness_predictor.py ML model
'''
import pickle, os
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np

# synthetic generator: features are counts of specific high-risk foods
foods = ["burger","pizza","raw_shellfish","spoiled_food","apple","banana","salad","fried_rice"]
def gen_sample():
    counts = [random.randint(0,5) for _ in foods]
    # risk label: 1 if raw_shellfish or spoiled_food present >0 or heavy fastfood
    risk = 1 if counts[foods.index("raw_shellfish")] > 0 or counts[foods.index("spoiled_food")] > 0 or (counts[foods.index("burger")] + counts[foods.index("pizza")] > 6) else 0
    return counts, risk

X = []
y = []
for _ in range(2000):
    xi, yi = gen_sample()
    X.append(xi); y.append(yi)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)
p = os.path.join(os.path.dirname(__file__), "..", "sickness_model.pkl")
with open(p,"wb") as f:
    pickle.dump(clf, f)
print("Saved", p)
'''
# # train_sickness_predictor.py
# import os
# import warnings
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# # -------------------- Suppress warnings -------------------- #
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')

# # -------------------- Paths -------------------- #
# BASE_DIR = "food-101/food-101"  # adjust if needed
# IMAGES_DIR = os.path.join(BASE_DIR, "images")
# TRAIN_DIR = os.path.join(BASE_DIR, "train")
# TEST_DIR = os.path.join(BASE_DIR, "test")
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sickness_cnn_model.h5")

# # -------------------- Parameters -------------------- #
# IMG_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS = 10
# LR = 1e-4

# # -------------------- Prepare data -------------------- #
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest",
#     validation_split=0.2  # split 20% for validation
# )

# train_gen = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',  # binary classification: sick vs safe
#     subset='training'
# )

# val_gen = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# # -------------------- Build CNN model -------------------- #
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model.trainable = False  # freeze pretrained weights

# x = GlobalAveragePooling2D()(base_model.output)
# x = Dropout(0.5)(x)

# # Binary output: 1 = high-risk, 0 = safe
# output = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=base_model.input, outputs=output)

# # -------------------- Compile -------------------- #
# model.compile(optimizer=Adam(learning_rate=LR),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # -------------------- Callbacks -------------------- #
# checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # -------------------- Train -------------------- #
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     callbacks=[checkpoint, reduce_lr, early_stop]
# )

# # -------------------- Cleanup -------------------- #
# tf.keras.backend.clear_session()
# print(f"✅ Training completed. CNN sickness model saved at '{MODEL_PATH}'")

# train_sickness_predictor.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------- Paths -------------------- #
MODEL_DIR = os.path.dirname(__file__)
SICKNESS_MODEL_PATH = os.path.join(MODEL_DIR, "..", "sickness_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "..", "food_labels.pkl")  # to save all possible foods

# -------------------- Example Food Labels -------------------- #
# Replace this with labels predicted by your CNN in real usage
food_labels = [
    "burger","pizza","raw_shellfish","spoiled_food",
    "apple","banana","salad","fried_rice","undercooked_meat"
]

# Save labels for later inference
with open(LABELS_PATH, "wb") as f:
    pickle.dump(food_labels, f)

NUM_FEATURES = len(food_labels)

# -------------------- Synthetic Dataset -------------------- #
# Here we simulate meals using food labels
# For production, replace this with actual CNN predictions per user history
def generate_sample():
    counts = np.random.randint(0, 5, size=NUM_FEATURES)
    # Sickness risk: if raw_shellfish, spoiled_food, or lots of fast food
    risk = 1 if counts[food_labels.index("raw_shellfish")] > 0 \
                or counts[food_labels.index("spoiled_food")] > 0 \
                or (counts[food_labels.index("burger")] + counts[food_labels.index("pizza")] > 6) \
            else 0
    return counts, risk

X, y = [], []
for _ in range(2000):
    xi, yi = generate_sample()
    X.append(xi)
    y.append(yi)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# -------------------- Build MLP -------------------- #
model = Sequential([
    Dense(64, activation='relu', input_shape=(NUM_FEATURES,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------- Callbacks -------------------- #
checkpoint = ModelCheckpoint(SICKNESS_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -------------------- Train -------------------- #
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stop],
    verbose=2
)

print(f"✅ Training completed. Sickness model saved at '{SICKNESS_MODEL_PATH}'")
