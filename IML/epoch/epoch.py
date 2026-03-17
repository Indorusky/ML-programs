import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np

# ---------- USER INPUT ----------
epochs_count = int(input("Enter number of epochs: "))

# ---------- DATA (Classification for accuracy) ----------
X = np.random.rand(100, 1)
y = (X > 0.5).astype(int)

# ---------- MODEL ----------
model = Sequential([
    Input(shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------- CUSTOM CALLBACK ----------
class PrintEpochStatus(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch+1} validated | "
            f"loss={logs['loss']:.4f}, "
            f"accuracy={logs['accuracy']:.4f}, "
            f"val_loss={logs['val_loss']:.4f}, "
            f"val_accuracy={logs['val_accuracy']:.4f}"
        )

    def on_train_end(self, logs=None):
        print(f"\nEarly stopping at epoch {self.model.stop_training_epoch}")

# ---------- EARLY STOPPING ----------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Patch to store stop epoch
def set_stop_epoch(self, epoch, logs=None):
    self.model.stop_training_epoch = epoch + 1

early_stop.on_epoch_end = set_stop_epoch.__get__(early_stop, EarlyStopping)

# ---------- TRAIN ----------
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=epochs_count,
    callbacks=[early_stop, PrintEpochStatus()],
    verbose=0
)

print("\nTraining completed")
print("Total epochs run:", len(history.history['loss']))
