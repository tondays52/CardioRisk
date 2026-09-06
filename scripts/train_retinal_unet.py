"""
CardioRisk AI - Rigorous Retinal Vessel U-Net Segmentation Training
Trained on Fundus-AVSeg Dataset with Combined Dice Loss + Focal/BCE Loss.
"""
import os
from typing import Any, List, Tuple, cast
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

layers = keras.layers

DATASET_ROOT = r"C:\Users\tonda\Desktop\dataset"
OUTPUT_DIRS = [
    r"C:\Users\tonda\Desktop\hrp\models",
    r"C:\Users\tonda\Desktop\hrp\models_scaled"
]
for out in OUTPUT_DIRS:
    os.makedirs(out, exist_ok=True)

IMG_SIZE = 128

avseg_dir = os.path.join(DATASET_ROOT, "Fundus-AVSeg", "Fundus-AVSeg")
train_txt = os.path.join(avseg_dir, 'training.txt')
test_txt = os.path.join(avseg_dir, 'testing.txt')

train_names: List[str] = []
test_names: List[str] = []

if os.path.exists(train_txt):
    with open(train_txt, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]

if os.path.exists(test_txt):
    with open(test_txt, 'r') as f:
        test_names = [line.strip() for line in f if line.strip()]


def load_data(names_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for nm in names_list:
        img_p = os.path.join(avseg_dir, 'images', nm)
        msk_p = os.path.join(avseg_dir, 'annotation', nm)
        if os.path.exists(img_p) and os.path.exists(msk_p):
            img = cv2.imread(img_p)
            msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
            if img is not None and msk is not None:
                # Resize
                img_res = cast(np.ndarray, cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                msk_res = cast(np.ndarray, cv2.resize(msk, (IMG_SIZE, IMG_SIZE)))
                
                # Green channel CLAHE enhancement for contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                green = img_res[:, :, 1]
                green_clahe = clahe.apply(green)
                
                # Stack 3-channel (RGB with enhanced green)
                img_enhanced = img_res.astype(np.float32) / 255.0
                img_enhanced[:, :, 1] = green_clahe.astype(np.float32) / 255.0
                
                msk_bin = (msk_res > 30).astype(np.float32)
                
                X.append(img_enhanced)
                y.append(msk_bin[..., np.newaxis])

    if not X:
        return (
            np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32),
            np.empty((0, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_train, y_train = load_data(train_names)
X_test, y_test = load_data(test_names)
print(f"Loaded Fundus-AVSeg: {len(X_train)} train images, {len(X_test)} test images")

if len(X_train) == 0:
    print("[WARNING] No training images loaded. Please verify dataset paths.")
    exit(0)

# Data Augmentation (Flips, 90-degree rotations)
X_aug_list: List[np.ndarray] = []
y_aug_list: List[np.ndarray] = []
for x, m in zip(X_train, y_train):
    X_aug_list.append(x)
    y_aug_list.append(m)
    X_aug_list.append(np.fliplr(x))
    y_aug_list.append(np.fliplr(m))
    X_aug_list.append(np.flipud(x))
    y_aug_list.append(np.flipud(m))
    X_aug_list.append(np.rot90(x, 1))
    y_aug_list.append(np.rot90(m, 1))

X_aug = np.array(X_aug_list, dtype=np.float32)
y_aug = np.array(y_aug_list, dtype=np.float32)
print(f"Augmented training set: {len(X_aug)} pairs")


# Register custom functions for seamless model saving / loading
@keras.saving.register_keras_serializable(package="CustomMetrics")
def dice_coef(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


@keras.saving.register_keras_serializable(package="CustomLosses")
def dice_bce_loss(y_true: Any, y_pred: Any) -> tf.Tensor:
    bce = keras.losses.binary_crossentropy(cast(Any, y_true), cast(Any, y_pred))
    dice = 1.0 - dice_coef(y_true, y_pred)
    return tf.cast(bce, tf.float32) + 1.5 * dice


def build_unet(input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 3)) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)

    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    return keras.Model(inputs, outputs, name="Retinal_Vessel_UNet")


model = build_unet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=dice_bce_loss,
    metrics=[dice_coef, 'accuracy']
)

model.fit(
    x=cast(Any, X_aug),
    y=cast(Any, y_aug),
    validation_data=(cast(Any, X_test), cast(Any, y_test)),
    epochs=25,
    batch_size=8,
    verbose=1
)

# Comprehensive Evaluation on Holdout Test Set
if len(X_test) > 0:
    test_preds = np.asarray(model.predict(cast(Any, X_test)))
    y_t = y_test.ravel()
    y_p = np.asarray(test_preds.ravel() > 0.5, dtype=np.float32)

    intersection = float(np.sum(y_t * y_p))
    dice = (2.0 * intersection) / (np.sum(y_t) + np.sum(y_p) + 1e-6)
    jaccard = intersection / (np.sum(y_t) + np.sum(y_p) - intersection + 1e-6)

    vessel_sensitivity = np.sum((y_t == 1) & (y_p == 1)) / (np.sum(y_t == 1) + 1e-6)
    vessel_specificity = np.sum((y_t == 0) & (y_p == 0)) / (np.sum(y_t == 0) + 1e-6)
    pixel_acc = float(np.mean(y_t == y_p))

    print("\n" + "=" * 60)
    print("RETINAL U-NET HOLDOUT TEST BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dice Coefficient (F1-Score): {dice:.4f} ({dice*100:.2f}%)")
    print(f"Jaccard Index (IoU)       : {jaccard:.4f} ({jaccard*100:.2f}%)")
    print(f"Vessel Sensitivity (Recall): {vessel_sensitivity*100:.2f}%")
    print(f"Background Specificity    : {vessel_specificity*100:.2f}%")
    print(f"Overall Pixel Accuracy    : {pixel_acc*100:.2f}%")

for out in OUTPUT_DIRS:
    p = os.path.join(out, "retinal_unet_model.keras")
    model.save(p)
    print(f"Saved Retinal U-Net to: {p}")
    p2 = os.path.join(out, "retinal_unet_scaled.keras")
    model.save(p2)
