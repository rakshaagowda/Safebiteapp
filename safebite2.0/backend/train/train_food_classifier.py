#just integrated with model
'''
import os
import warnings

# 🧹 Clean up TensorFlow/NumPy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 🧠 Silence TensorFlow logs and OMP noise
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ["OMP_NUM_THREADS"] = "1"

# ✅ Import TensorFlow after setting env vars
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')  # <-- put it right here ✅

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: disable oneDNN custom ops warning
tf.get_logger().setLevel('ERROR')

# Paths
BASE_DIR = "food-101"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

# ✅ Data preprocessing
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

# ✅ Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base layers for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Callbacks
checkpoint = ModelCheckpoint("food_classifier.h5", save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop]
)
tf.keras.backend.clear_session()


print("✅ Training completed. Model saved as 'food_classifier.h5'")
'''

#before splitting images into train and test
'''
import os
import warnings
import tensorflow as tf
import shutil
import random
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
#BASE_DIR = "food-101"  # Make sure dataset is downloaded and extracted here
#TRAIN_DIR = os.path.join(BASE_DIR, "train")
#TEST_DIR = os.path.join(BASE_DIR, "test")


BASE_DIR = "food-101/food-101"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")


# -------------------- Parameters -------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

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
base_model.trainable = False  # Freeze base layers for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# -------------------- Compile -------------------- #
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- Callbacks -------------------- #
checkpoint = ModelCheckpoint("food_classifier.h5", save_best_only=True, monitor='val_accuracy', mode='max')
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

print("✅ Training completed. Model saved as 'food_classifier.h5'")
'''
import os
import warnings
import tensorflow as tf
import shutil
import random
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
BASE_DIR = "food-101/food-101"  # Your current folder structure
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# -------------------- Create train/test split if not exist -------------------- #
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    print("🟢 Creating train/test directories and splitting dataset...")

    # Remove old empty folders if they exist
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Only directories (ignore hidden files like .DS_Store)
    categories = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]

    for category in categories:
        os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

        imgs = os.listdir(os.path.join(IMAGES_DIR, category))
        random.shuffle(imgs)
        split_idx = int(0.8 * len(imgs))

        print(f"➡️ Processing category: {category} ({len(imgs)} images)")

        for i, img in enumerate(imgs[:split_idx]):
            shutil.copy(os.path.join(IMAGES_DIR, category, img),
                        os.path.join(TRAIN_DIR, category, img))
            if (i + 1) % 50 == 0 or i == split_idx - 1:
                print(f"   Copied {i+1}/{split_idx} images to train")

        for i, img in enumerate(imgs[split_idx:]):
            shutil.copy(os.path.join(IMAGES_DIR, category, img),
                        os.path.join(TEST_DIR, category, img))
            if (i + 1) % 50 == 0 or i == len(imgs) - split_idx - 1:
                print(f"   Copied {i+1}/{len(imgs) - split_idx} images to test")

    print("✅ Dataset split complete.")

# -------------------- Parameters -------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

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
base_model.trainable = False  # Freeze base layers for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# -------------------- Compile -------------------- #
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- Callbacks -------------------- #
checkpoint = ModelCheckpoint("food_classifier.h5", save_best_only=True, monitor='val_accuracy', mode='max')
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

print("✅ Training completed. Model saved as 'food_classifier.h5'")
