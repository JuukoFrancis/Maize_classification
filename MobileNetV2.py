"""
=============================================================
  MobileNetV2 - Maize Leaf Disease Classification
  TRAINING SCRIPT
  Classes: MLV (Maize Leaf Virus), MSV (Maize Streak Virus), Healthy
=============================================================

FOLDER STRUCTURE EXPECTED:
    dataset/
    ├── train/
    │   ├── MLV/        (~2000 images)
    │   ├── MSV/        (~2000 images)
    │   └── Healthy/    (~2500 images)
    └── val/
        ├── MLV/        (~400 images)
        ├── MSV/        (~400 images)
        └── Healthy/    (~400 images)

OUTPUT FILES (saved in same folder as this script):
    - mobilenetv2_maize_model.h5       (trained model)
    - mobilenetv2_training_history.png (accuracy & loss curves)
    - mobilenetv2_training_log.csv     (epoch-by-epoch metrics)
=============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # saves graphs even without a display
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# Import the shared graph helper
from plot_training_graphs import save_graphs

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CUSTOM EARLY STOPPING AT TARGET TRAIN ACCURACY
# ─────────────────────────────────────────────
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_acc=0.90):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        if acc is not None and acc >= self.target_acc:
            print(f"\n Target training accuracy {self.target_acc*100:.2f}% reached. Stopping training.")
            self.model.stop_training = True

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_DIR      = "dataset/train"
VAL_DIR        = "dataset/val"
MODEL_SAVE     = "mobilenetv2_maize_model.h5"
HISTORY_PLOT   = "mobilenetv2_training_history.png"
LOG_CSV        = "mobilenetv2_training_log.csv"

IMG_SIZE       = (224, 224)
BATCH_SIZE     = 16
EPOCHS         = 50
LEARNING_RATE  = 0.0001
DROPOUT_RATE   = 0.2
NUM_CLASSES    = 3
CLASS_NAMES    = ["Healthy", "MLV", "MSV"]

# ─────────────────────────────────────────────
# 2.  DATA GENERATORS
# ─────────────────────────────────────────────
print("\n[1/5] Loading datasets...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"\n  Classes found : {train_generator.class_indices}")
print(f"  Train samples : {train_generator.samples}")
print(f"  Val samples   : {val_generator.samples}")

# ─────────────────────────────────────────────
# 3.  CLASS WEIGHTS
# ─────────────────────────────────────────────
total = train_generator.samples
class_counts = np.bincount(train_generator.classes)
class_weight_dict = {}
for i, count in enumerate(class_counts):
    class_weight_dict[i] = total / (NUM_CLASSES * count)
print(f"\n  Class weights : {class_weight_dict}")

# ─────────────────────────────────────────────
# 4.  BUILD MODEL
# ─────────────────────────────────────────────
print("\n[2/5] Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(DROPOUT_RATE)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"\n  Total params    : {model.count_params():,}")
print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ─────────────────────────────────────────────
# 5.  CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_CSV),
    StopAtAccuracy(target_acc=0.94),
]

# ─────────────────────────────────────────────
# 6.  PHASE 1 TRAINING  (frozen base)
# ─────────────────────────────────────────────
print("\n[3/5] Phase 1: Training classification head (base frozen)...")

history1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

phase2_start_epoch = len(history1.history["accuracy"]) + 1   # ← remember for graph

# ─────────────────────────────────────────────
# 7.  PHASE 2 FINE-TUNING
# ─────────────────────────────────────────────
print("\n[4/5] Phase 2: Fine-tuning (unfreezing last 30 layers)...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_ft = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1),
    StopAtAccuracy(target_acc=0.94),
    CSVLogger(LOG_CSV, append=True),
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_ft,
    class_weight=class_weight_dict,
    verbose=1
)

# ─────────────────────────────────────────────
# 8.  SAVE TRAINING GRAPHS
# ─────────────────────────────────────────────
print("\n[5/5] Saving training curves...")

acc      = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss     = history1.history["loss"]         + history2.history["loss"]
val_loss = history1.history["val_loss"]     + history2.history["val_loss"]

save_graphs(
    acc=acc,
    val_acc=val_acc,
    loss=loss,
    val_loss=val_loss,
    title="MobileNetV2 – Maize Disease Classification",
    out_path=HISTORY_PLOT,
    phase2_start_epoch=phase2_start_epoch,
)

# ─────────────────────────────────────────────
# 9.  FINAL SUMMARY
# ─────────────────────────────────────────────
best_val_acc = max(val_acc)
best_epoch   = val_acc.index(best_val_acc) + 1

print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%  (Epoch {best_epoch})")
print(f"  Model saved to           : {MODEL_SAVE}")
print(f"  Training log saved to    : {LOG_CSV}")
print(f"  Curves saved to          : {HISTORY_PLOT}")
print("="*55)
