import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths for dataset and model
BASE_PATH = r"D:\project\DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI-main\Code\data"
MODEL_PATH = r"D:\project\DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI-main\Code\model\vit_deepfake_model.h5"

# Prepare dataset using ImageDataGenerator
def prepare_dataset():
    train_dir = os.path.join(BASE_PATH, "train")
    val_dir = os.path.join(BASE_PATH, "val")

    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_dataset = datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )
    val_dataset = datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )

    return train_dataset, val_dataset

# Build the Vision Transformer model
def build_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Train the model
def train_model():
    train_dataset, val_dataset = prepare_dataset()

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        verbose=1
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_model()
