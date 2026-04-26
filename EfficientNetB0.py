"""
=============================================================
  EfficientNetB0 - Maize Leaf Disease Classification
  TRAINING SCRIPT
  Classes: MLV (Maize Leaf Virus), MSV (Maize Streak Virus), Healthy
=============================================================

OUTPUT FILES:
    - best_phase1.keras                 (best model from phase 1)
    - best_phase2.keras / efficientnetb0_maize_finetuned.keras
    - efficientnetb0_training_history.png  (accuracy & loss curves)
    - efficientnetb0_training_log.csv
=============================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import the shared graph helper
from plot_training_graphs import save_graphs

# ── Custom Early Stopping  ─────────────────────────────────
class StopAt92(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy", 0) >= 0.92:
            print(f"\n   Reached 92% validation accuracy at epoch {epoch + 1}. Stopping.")
            self.model.stop_training = True

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 3
TRAIN_DIR   = "dataset/train"
VAL_DIR     = "dataset/val"
LOG_CSV     = "efficientnetb0_training_log.csv"
HISTORY_PLOT = "efficientnetb0_training_history.png"

# ── Data ──────────────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"Classes: {train_data.class_indices}")

# ── Class weights ─────────────────────────────────────────────────────────────
total = train_data.samples
class_counts = np.bincount(train_data.classes)

weights = {}
for i, count in enumerate(class_counts):
    weights[i] = total / (NUM_CLASSES * count)

print(weights)

# ── Model ─────────────────────────────────────────────────────────────────────
base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = models.Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── Phase 1 callbacks ─────────────────────────────────────────────────────────
callbacks_p1 = [
    StopAt92(),
    ModelCheckpoint("best_phase1.keras", save_best_only=True,
                    monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True,
                  monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    CSVLogger(LOG_CSV),
]

print("\n── Phase 1: training head only ──")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=weights,
    callbacks=callbacks_p1
)

phase2_start_epoch = len(history1.history["accuracy"]) + 1   # ← for graph line

# ── Phase 2: fine-tune top ~30% of base ──────────────────────────────────────
base.trainable = True
fine_tune_from = int(len(base.layers) * 0.7)
for layer in base.layers[:fine_tune_from]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    StopAt92(),
    ModelCheckpoint("best_phase2.keras", save_best_only=True,
                    monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=7, restore_best_weights=True,
                  monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_CSV, append=True),
]

print("\n── Phase 2: fine-tuning top layers ──")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    class_weight=weights,
    callbacks=callbacks_p2
)

# ── Save model ────────────────────────────────────────────────────────────────
model.save("efficientnetb0_maize_finetuned.keras")
print("Model saved.")

# ── Save training graphs ──────────────────────────────────────────────────────
acc      = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss     = history1.history["loss"]         + history2.history["loss"]
val_loss = history1.history["val_loss"]     + history2.history["val_loss"]

save_graphs(
    acc=acc,
    val_acc=val_acc,
    loss=loss,
    val_loss=val_loss,
    title="EfficientNetB0 – Maize Disease Classification",
    out_path=HISTORY_PLOT,
    phase2_start_epoch=phase2_start_epoch,
)

best_val_acc = max(val_acc)
best_epoch   = val_acc.index(best_val_acc) + 1

print("\n" + "="*55)
print("  TRAINING COMPLETE — EfficientNetB0")
print("="*55)
print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%  (Epoch {best_epoch})")
print(f"  Training log saved to    : {LOG_CSV}")
print(f"  Curves saved to          : {HISTORY_PLOT}")
print("="*55)