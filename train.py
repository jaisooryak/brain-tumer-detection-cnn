import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# Enable mixed precision using tf.keras API
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (200, 200)
BATCH_SIZE  = 64
EPOCHS_P1   = 10   # Phase 1: frozen base (feature extraction)
EPOCHS_P2   = 15   # Phase 2: fine-tuning top layers
AUTOTUNE    = tf.data.AUTOTUNE

train_path = "dataset/Training"
test_path  = "dataset/Testing"

# ── Load dataset ─────────────────────────────────────────────────────────────
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_data.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ── Augmentation (stronger) ──────────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="augmentation")

# ── Performance ──────────────────────────────────────────────────────────────
train_data = train_data.prefetch(AUTOTUNE)
test_data  = test_data.prefetch(AUTOTUNE)

# ── Build model ──────────────────────────────────────────────────────────────
def build_model(trainable_base=False, unfreeze_from=None):
    """
    Phase 1 → trainable_base=False  (train head only)
    Phase 2 → trainable_base=True, unfreeze_from=-30  (fine-tune top 30 layers)
    """
    # EfficientNetB3 — significantly stronger than MobileNetV2
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )

    if not trainable_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if unfreeze_from is not None:
            # Freeze everything except the last N layers
            for layer in base_model.layers[:unfreeze_from]:
                layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    x = data_augmentation(inputs)

    # EfficientNet has its own preprocessing built-in (no separate call needed)
    x = base_model(x, training=False)

    # Stronger classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Train head only (frozen base)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("PHASE 1: Feature Extraction (frozen base)")
print("="*50)

model = build_model(trainable_base=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=4,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=2, min_lr=1e-6, verbose=1
    ),
]

history_p1 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS_P1,
    callbacks=callbacks_p1,
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Fine-tune top layers of EfficientNetB3
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("PHASE 2: Fine-Tuning (top 30 layers unfrozen)")
print("="*50)

# Unfreeze the top 30 layers of the base
base = model.layers[2]   # EfficientNetB3 is the 3rd layer (after input + augment)
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

# Much lower LR to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=3, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
]

history_p2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS_P2,
    callbacks=callbacks_p2,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(test_data, verbose=0)
print(f"\n✅ Final Test Accuracy: {accuracy*100:.2f}%")
print(f"   Final Test Loss    : {loss:.4f}")

# ── Merge histories for plotting ──────────────────────────────────────────────
def merge_history(h1, h2, key):
    return h1.history[key] + h2.history[key]

acc      = merge_history(history_p1, history_p2, 'accuracy')
val_acc  = merge_history(history_p1, history_p2, 'val_accuracy')
loss_h   = merge_history(history_p1, history_p2, 'loss')
val_loss = merge_history(history_p1, history_p2, 'val_loss')
phase_boundary = len(history_p1.history['accuracy'])

# ── Plot Accuracy ─────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(acc,     label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tune start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Phase 1 + 2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── Plot Loss ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(loss_h,   label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tune start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch (Phase 1 + 2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("\nGenerating confusion matrix...")

y_true, y_pred = [], []
for images, labels in test_data:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("brain_tumor_model_v2.keras")
print("\n✅ Model saved as brain_tumor_model_v2.keras")