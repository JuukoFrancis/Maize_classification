"""
=============================================================
  ResNet50 - Maize Leaf Disease Classification
  TRAINING SCRIPT
  Classes: MLV (Maize Leaf Virus), MSV (Maize Streak Virus), Healthy
=============================================================

OUTPUT FILES:
    - resnet50_best_phase1.keras
    - resnet50_best_phase2.keras / resnet50_maize_final.keras
    - resnet50_training_history.png    (accuracy & loss curves)
    - resnet50_training_log.csv
=============================================================
"""

# Suppresses Tensorflow log warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")  # important libraries used like tensorflow, numpy

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    Callback,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the shared graph helper
from plot_training_graphs import save_graphs


# ── Custom Early Stopping at 91% val_accuracy ─────────────────────────────────
class StopAt91(
    Callback
):  # creates our own callback , then checks if the validation accuracy is >=90 if true it stops training.
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy", 0) >= 0.90:
            print(
                f"\n  Reached 90.0% validation accuracy at epoch {epoch + 1}. Stopping."
            )
            self.model.stop_training = True


# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)  # Image size used by resnet
BATCH_SIZE = 32  # No of images per training step
NUM_CLASSES = 3  # MSV. MLB ,HEALTHY
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
LOG_CSV = "resnet50_training_log.csv"
HISTORY_PLOT = "resnet50_training_history.png"

# ── DATA PREPROCESSING/DATA AUGMENTATION IS DONE to prevent overfitting-------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalizes the image for our model & removes variations
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)  # THE FEATURES ABOVE ARE APPLIED TO THE IMAGES

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)  # validation data is not augmentation

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)  # THIS LOADS OUR IMAGES FROM THE TRAIN DIRECTORY/FOLDER.

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,  # keeps the validation shuffle order consistent
)  # THIS LOADS OUR IMAGES FROM THE VALIDATION DIRECTORY/FOLDER.

print(
    f"Classes: {train_data.class_indices}"
)  # Shows mapping in this 'Healthy': 0, 'MLV': 1, 'MSV': 2

# ── Class weights ─────────────────────────────────────────────────────────────
# total = 6500
# weights = {
#     0: total / (NUM_CLASSES * 2500),   # healthy
#     1: total / (NUM_CLASSES * 2000),   # mlv
#     2: total / (NUM_CLASSES * 2000),   # msv
# }
# print(f"Class weights: {weights}")

total = train_data.samples
class_counts = np.bincount(train_data.classes)

weights = {}  # This assigns the assigns importance to classes and handles imbalance
for i, count in enumerate(class_counts):
    weights[i] = total / (NUM_CLASSES * count)

print(weights)

# ── Build model ───────────────────────────────────────────────────────────────
base = ResNet50(
    weights="imagenet",  # This is loading the pretrained model and removes the final classifier
    include_top=False,
    input_shape=(224, 224, 3),
)

# ── Phase 1: freeze entire base ───────────────────────────────────────────────
base.trainable = False  # This Freezes the base model and helps in feature extraction
# Freezing is used to keep pretrained features unchanged during initial training
x = base.output  # Get the extracted features
x = layers.GlobalAveragePooling2D()(x)  # make extracted features averaged.
x = layers.BatchNormalization()(x)  # Normalization that stabilizes learning
x = layers.Dense(  # This is like our fully connected layer with our activation function RELU
    256,
    activation="relu",  # easier
    kernel_regularizer=regularizers.l2(1e-4),  # reduces learning rate
)(
    x
)
x = layers.Dropout(0.4)(
    x
)  # Prevents overfitting/ gets all specific unique features/ helps remove redundant neurons
output = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(
    x
)  # This the final output with softmax probabilities

model = models.Model(
    inputs=base.input, outputs=output
)  # THIS BUILD THE MODEL BY CONNECTING THE INPUT BASE TO THE OUTPUT BASE

model.compile(
    optimizer=Adam(learning_rate=1e-3),  # THIS UPDATES THE WEIGHT WHEN TRAINING
    loss="categorical_crossentropy",  # MEASURES PREDICTION ERROR
    metrics=["accuracy"],  # MEASURES CORRECTNESS
)

model.summary()

callbacks_p1 = [
    StopAt91(),
    ModelCheckpoint(  # SAVES THE BEST MODEL
        "resnet50_best_phase1.keras",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1,
    ),
    EarlyStopping(  # STOPS IF THERE IS NO MOVEMENT
        patience=5, restore_best_weights=True, monitor="val_accuracy", verbose=1
    ),
    ReduceLROnPlateau(  # REDUCES THE LEARNING RATE
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
    ),
    CSVLogger(LOG_CSV),  # SAVES OUR TRAINING LOGS
]

print("\n" + "=" * 50)
print("   PHASE 1: Training head only")
print("=" * 50)
history1 = model.fit(  # THIS TRAINS ONLY THE CLASSIFIER
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=weights,
    callbacks=callbacks_p1,
)

phase2_start_epoch = len(history1.history["accuracy"]) + 1  # ← for graph line

# ── Phase 2: unfreeze top 30% of ResNet50 ────────────────────────────────────
base.trainable = True  # UNFREEZES THE MODEL WHICH allows the model to adapt those features to the specific maize disease dataset through fine-tuning.
fine_tune_from = int(len(base.layers) * 0.7)  # UNFREEZES THE TOP 30%

for layer in base.layers[:fine_tune_from]:  # KEEPS LOWER LAYERS FROZEN
    layer.trainable = False

# Keep BatchNorm layers frozen to preserve learned statistics
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(  # RECOMPILE AND REQUIRED WHEN CHANGING LAYERS
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"\nTrainable layers in phase 2: {trainable_count}")

callbacks_p2 = [
    StopAt91(),
    ModelCheckpoint(
        "resnet50_best_phase2.keras",
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1,
    ),
    EarlyStopping(
        patience=7, restore_best_weights=True, monitor="val_accuracy", verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
    ),
    CSVLogger(LOG_CSV, append=True),
]

print("\n" + "=" * 50)
print("   PHASE 2: Fine-tuning top layers")
print("=" * 50)
history2 = model.fit(  # FINE TUNING THE MODEL
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=weights,
    callbacks=callbacks_p2,
)

# ── Save final model ──────────────────────────────────────────────────────────
model.save("resnet50_maize_final.keras")  # SAVES THE  TRAINED MODEL
print("\nFinal model saved as resnet50_maize_final.keras")

# ── Save training graphs ──────────────────────────────────────────────────────
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

save_graphs(
    acc=acc,
    val_acc=val_acc,
    loss=loss,
    val_loss=val_loss,
    title="ResNet50 – Maize Disease Classification",
    out_path=HISTORY_PLOT,
    phase2_start_epoch=phase2_start_epoch,
)

best_val_acc = max(val_acc)
best_epoch = val_acc.index(best_val_acc) + 1

print("\n" + "=" * 55)
print("  TRAINING COMPLETE — ResNet50")
print("=" * 55)
print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%  (Epoch {best_epoch})")
print(f"  Training log saved to    : {LOG_CSV}")
print(f"  Curves saved to          : {HISTORY_PLOT}")
print("=" * 55)
