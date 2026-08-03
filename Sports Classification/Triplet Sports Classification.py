"""
CAB420 Project - Sports Image Classification
Method: Triplet Network / Deep Metric Learning
Author: Fan-Bo Kong method section

What this script does:
1. Loads the sports CSV file.
2. Ignores the original Kaggle train/valid/test split.
3. Creates a new stratified 80/10/10 split.
4. Trains a Triplet Network using a MobileNetV2 embedding backbone.
5. Evaluates classification by comparing test embeddings to train class centroids.
6. Reports Top-1 Accuracy, Top-5 Accuracy, Macro F1, Classification Report, and Confusion Matrix.

Expected CSV columns:
- class id
- filepaths
- labels
- data set

Expected image folder structure:
DATA_ROOT/
    train/air hockey/001.jpg
    valid/air hockey/1.jpg
    test/air hockey/1.jpg
    ...

Important:
Change CSV_PATH and DATA_ROOT before running.
"""

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    top_k_accuracy_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Change these paths to match your setup.
# In Colab, you might use something like:
# CSV_PATH = "/content/sports.csv"
# DATA_ROOT = "/content/100-sports-image-classification"
# In Kaggle, DATA_ROOT is usually under /kaggle/input/...
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "archive" / "sports.csv"
DATA_ROOT = BASE_DIR / "archive"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EMBEDDING_DIM = 128
MARGIN = 0.3
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 5
LEARNING_RATE_STAGE_1 = 1e-4
LEARNING_RATE_STAGE_2 = 1e-5
OUTPUT_DIR = "triplet_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load CSV and create 80/10/10 stratified split
# ============================================================

def is_readable_image(path):
    """Return True only if the file exists and PIL can read it as an image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return False


def load_and_resplit(csv_path, data_root):
    df = pd.read_csv(csv_path)

    required_cols = {"class id", "filepaths", "labels", "data set"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Make full image path. This keeps the original folders; it only changes the split in the dataframe.
    df["path"] = df["filepaths"].apply(lambda p: os.path.join(str(data_root), str(p).replace("\\", "/")))

    # Remove missing or corrupted images before splitting/training.
    # Some Kaggle image datasets contain a small number of unreadable files.
    exists_mask = df["path"].apply(os.path.exists)
    missing_df = df.loc[~exists_mask].copy()
    if len(missing_df) > 0:
        missing_csv = os.path.join(OUTPUT_DIR, "missing_images.csv")
        missing_df.to_csv(missing_csv, index=False)
        print(f"WARNING: Removed {len(missing_df)} missing image files.")
        print(f"Missing image list saved to: {missing_csv}")
        print("Example missing path:", missing_df["path"].iloc[0])

    df = df.loc[exists_mask].reset_index(drop=True)

    readable_mask = df["path"].apply(is_readable_image)
    bad_df = df.loc[~readable_mask].copy()
    if len(bad_df) > 0:
        bad_csv = os.path.join(OUTPUT_DIR, "unreadable_images.csv")
        bad_df.to_csv(bad_csv, index=False)
        print(f"WARNING: Removed {len(bad_df)} corrupted/unreadable image files.")
        print(f"Unreadable image list saved to: {bad_csv}")
        print("Example bad path:", bad_df["path"].iloc[0])

    df = df.loc[readable_mask].reset_index(drop=True)

    # Encode labels from text to 0..99. This avoids relying on the original class id ordering.
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["labels"])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    # First split: 90% train+valid, 10% test.
    train_valid_df, test_df = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label_id"],
        random_state=SEED,
        shuffle=True,
    )

    # Second split: from remaining 90%, take 1/9 as validation.
    # 1/9 of 90% = 10%, giving final 80/10/10.
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=1 / 9,
        stratify=train_valid_df["label_id"],
        random_state=SEED,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save split files so the exact split can be reused by your group.
    train_df.to_csv(os.path.join(OUTPUT_DIR, "sports_train_80.csv"), index=False)
    valid_df.to_csv(os.path.join(OUTPUT_DIR, "sports_valid_10.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "sports_test_10.csv"), index=False)

    print("Original split from CSV:")
    print(df["data set"].value_counts())
    print("\nNew split sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test : {len(test_df)}")
    print(f"Classes: {num_classes}")

    return train_df, valid_df, test_df, label_encoder, class_names, num_classes


# ============================================================
# 3. Triplet batch generator
# ============================================================

class TripletSequence(keras.utils.Sequence):
    """
    Generates batches of anchor, positive and negative images.

    Anchor:   an image from class A
    Positive: a different image from class A
    Negative: an image from class B, where B != A
    """

    def __init__(self, df, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        self.paths = self.df["path"].values
        self.labels = self.df["label_id"].values.astype(np.int32)
        self.indices = np.arange(len(self.df))

        self.class_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }
        self.classes = np.array(sorted(self.class_to_indices.keys()))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, path):
        img = load_img(path, target_size=self.image_size, color_mode="rgb")
        img = img_to_array(img).astype("float32")
        # Keep 0-255 values because MobileNetV2 preprocessing is inside the model.
        return img

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        current_batch_size = len(batch_indices)

        anchors = np.zeros((current_batch_size, *self.image_size, 3), dtype="float32")
        positives = np.zeros((current_batch_size, *self.image_size, 3), dtype="float32")
        negatives = np.zeros((current_batch_size, *self.image_size, 3), dtype="float32")

        for i, anchor_idx in enumerate(batch_indices):
            anchor_label = self.labels[anchor_idx]

            # Positive: same class, preferably not the same exact image.
            positive_candidates = self.class_to_indices[anchor_label]
            if len(positive_candidates) > 1:
                positive_idx = anchor_idx
                while positive_idx == anchor_idx:
                    positive_idx = np.random.choice(positive_candidates)
            else:
                positive_idx = anchor_idx

            # Negative: different class.
            negative_label = anchor_label
            while negative_label == anchor_label:
                negative_label = np.random.choice(self.classes)
            negative_idx = np.random.choice(self.class_to_indices[negative_label])

            anchors[i] = self._load_image(self.paths[anchor_idx])
            positives[i] = self._load_image(self.paths[positive_idx])
            negatives[i] = self._load_image(self.paths[negative_idx])

        # y is not used, but Keras Sequence works cleanly when x and y are returned.
        dummy_y = np.zeros((current_batch_size,), dtype="float32")
        return (anchors, positives, negatives), dummy_y


# ============================================================
# 4. Build Triplet Network
# ============================================================

def build_embedding_model(num_trainable_base_layers=0):
    """
    Creates the image embedding network.
    The output is a 128-D L2-normalised embedding vector.
    """

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze all base layers first.
    base_model.trainable = False

    # Optionally unfreeze only the last few layers for fine-tuning.
    if num_trainable_base_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-num_trainable_base_layers]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(EMBEDDING_DIM)(x)

    # L2 normalisation makes distance comparisons more stable.
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_embedding")(x)

    return keras.Model(inputs, outputs, name="embedding_model")


class DistanceLayer(layers.Layer):
    """Computes anchor-positive and anchor-negative squared distances."""

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return ap_distance, an_distance


def build_siamese_network(embedding_model):
    anchor_input = keras.Input(shape=(*IMAGE_SIZE, 3), name="anchor")
    positive_input = keras.Input(shape=(*IMAGE_SIZE, 3), name="positive")
    negative_input = keras.Input(shape=(*IMAGE_SIZE, 3), name="negative")

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    distances = DistanceLayer(name="distance_layer")(
        anchor_embedding,
        positive_embedding,
        negative_embedding,
    )

    return keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="triplet_network",
    )


class TripletModel(keras.Model):
    def __init__(self, siamese_network, margin=MARGIN):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.siamese_network(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _compute_loss(self, data, training):
        ap_distance, an_distance = self.siamese_network(data, training=training)
        loss = ap_distance - an_distance + self.margin
        loss = tf.maximum(loss, 0.0)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        # Keras passes (x, y). We only need x.
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data, training=True)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        loss = self._compute_loss(data, training=False)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# ============================================================
# 5. Evaluation helpers
# ============================================================

def load_eval_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def make_eval_dataset(df, batch_size=BATCH_SIZE):
    paths = df["path"].values
    labels = df["label_id"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_eval_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def extract_embeddings(embedding_model, df):
    ds = make_eval_dataset(df)
    image_ds = ds.map(lambda images, labels: images)
    embeddings = embedding_model.predict(image_ds, verbose=1)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in ds], axis=0)
    return embeddings, labels


def compute_class_centroids(train_embeddings, train_labels, num_classes):
    centroids = []
    for class_id in range(num_classes):
        class_embeddings = train_embeddings[train_labels == class_id]
        centroid = class_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)
    return np.vstack(centroids)


def evaluate_with_centroids(embedding_model, train_df, test_df, class_names, num_classes):
    start_time = time.time()

    print("\nExtracting train embeddings...")
    train_embeddings, train_labels = extract_embeddings(embedding_model, train_df)

    print("\nExtracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(embedding_model, test_df)

    centroids = compute_class_centroids(train_embeddings, train_labels, num_classes)

    # Because embeddings and centroids are L2-normalised, dot product is cosine similarity.
    similarity_scores = np.matmul(test_embeddings, centroids.T)
    y_pred = np.argmax(similarity_scores, axis=1)

    top1_acc = accuracy_score(test_labels, y_pred)
    top5_acc = top_k_accuracy_score(
        test_labels,
        similarity_scores,
        k=5,
        labels=np.arange(num_classes),
    )
    macro_f1 = f1_score(test_labels, y_pred, average="macro")

    inference_time = time.time() - start_time

    print("\n================ Test Results ================")
    print(f"Top-1 Accuracy / Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy:           {top5_acc:.4f}")
    print(f"Macro F1 Score:           {macro_f1:.4f}")
    print(f"Embedding + evaluation time: {inference_time:.2f} seconds")

    report = classification_report(
        test_labels,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
        f.write(f"Top-1 Accuracy / Accuracy: {top1_acc:.4f}\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Embedding + evaluation time: {inference_time:.2f} seconds\n")

    cm = confusion_matrix(test_labels, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(28, 28))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    plt.title("Triplet Network Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_triplet.png"), dpi=200)
    plt.close()

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "inference_time_seconds": inference_time,
        "y_true": test_labels,
        "y_pred": y_pred,
        "similarity_scores": similarity_scores,
    }


def plot_training_history(history, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet loss")
    plt.title("Triplet Network Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


# ============================================================
# 6. Main training pipeline
# ============================================================

def main():
    train_df, valid_df, test_df, label_encoder, class_names, num_classes = load_and_resplit(
        CSV_PATH,
        DATA_ROOT,
    )

    train_sequence = TripletSequence(train_df, batch_size=BATCH_SIZE, shuffle=True)
    valid_sequence = TripletSequence(valid_df, batch_size=BATCH_SIZE, shuffle=False)

    # Stage 1: train only the embedding head while MobileNetV2 is frozen.
    embedding_model = build_embedding_model(num_trainable_base_layers=0)
    siamese_network = build_siamese_network(embedding_model)
    triplet_model = TripletModel(siamese_network, margin=MARGIN)
    triplet_model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE_STAGE_1))

    # Do not checkpoint the custom TripletModel directly.
    # Keras cannot serialize this subclassed model without a custom get_config().
    # EarlyStopping restores the best weights in memory, then we save the embedding_model below.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    print("\nStarting Stage 1 training: frozen MobileNetV2 backbone")
    start_train = time.time()
    history_stage_1 = triplet_model.fit(
        train_sequence,
        validation_data=valid_sequence,
        epochs=EPOCHS_STAGE_1,
        callbacks=callbacks,
    )
    stage_1_time = time.time() - start_train
    print(f"Stage 1 training time: {stage_1_time:.2f} seconds")
    plot_training_history(history_stage_1, "triplet_loss_stage1.png")

    # Save the embedding model after Stage 1.
    embedding_model.save(os.path.join(OUTPUT_DIR, "embedding_model_stage1.keras"))

    # Stage 2: optional fine-tuning of the last MobileNetV2 layers.
    # This usually improves embeddings but takes longer.
    print("\nStarting Stage 2 training: fine-tune last MobileNetV2 layers")
    embedding_model_ft = build_embedding_model(num_trainable_base_layers=30)

    # Transfer learned weights from Stage 1 where layer names/shapes match.
    # If this gives warnings, it is normally safe; the architecture is the same except trainable flags.
    embedding_model_ft.set_weights(embedding_model.get_weights())

    siamese_network_ft = build_siamese_network(embedding_model_ft)
    triplet_model_ft = TripletModel(siamese_network_ft, margin=MARGIN)
    triplet_model_ft.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE_STAGE_2))

    # Same issue here: checkpoint the embedding model only after training,
    # rather than trying to serialize the custom TripletModel wrapper.
    callbacks_ft = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    start_ft = time.time()
    history_stage_2 = triplet_model_ft.fit(
        train_sequence,
        validation_data=valid_sequence,
        epochs=EPOCHS_STAGE_2,
        callbacks=callbacks_ft,
    )
    stage_2_time = time.time() - start_ft
    print(f"Stage 2 fine-tuning time: {stage_2_time:.2f} seconds")
    plot_training_history(history_stage_2, "triplet_loss_stage2.png")

    total_training_time = stage_1_time + stage_2_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

    embedding_model_ft.save(os.path.join(OUTPUT_DIR, "embedding_model_final.keras"))

    results = evaluate_with_centroids(
        embedding_model_ft,
        train_df,
        test_df,
        class_names,
        num_classes,
    )

    with open(os.path.join(OUTPUT_DIR, "summary_results.txt"), "w", encoding="utf-8") as f:
        f.write("CAB420 Sports Triplet Network Results\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(valid_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        f.write(f"Top-1 Accuracy / Accuracy: {results['top1_accuracy']:.4f}\n")
        f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}\n")
        f.write(f"Macro F1 Score: {results['macro_f1']:.4f}\n")
        f.write(f"Embedding + evaluation time: {results['inference_time_seconds']:.2f} seconds\n")

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
