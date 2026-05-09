"""
src/data_loader.py
==================
Dataset ingestion and preprocessing pipeline for the Intel Image Classification
dataset. Implements a tf.data-based input pipeline with:
  - Directory-based class discovery
  - Deterministic train/val stratified split from the training directory
  - Runtime data augmentation (training split only)
  - Prefetching and caching for GPU throughput optimization

All configuration is read from configs/config.yaml to ensure a single source
of truth for all hyperparameters.
"""

import os
import random
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility: Load configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Parse the project YAML configuration file.

    Args:
        config_path: Path to the YAML config, relative to project root.

    Returns:
        Nested dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Utility: Seed all random number generators for reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducibility.

    Note: Full determinism across GPU runs is not guaranteed due to
    non-deterministic CUDA kernels; however, this ensures consistency
    across CPU-bound operations.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Core: Image loading function (used inside tf.data.Dataset.map)
# ---------------------------------------------------------------------------

def _load_and_preprocess_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    img_height: int,
    img_width: int
) -> tuple:
    """Read, decode, resize, and normalise a single image.

    This function is designed to be called inside a tf.data pipeline via
    `dataset.map()`. It reads the raw bytes from disk, decodes the JPEG/PNG,
    resizes to the target resolution, and scales pixel values to [0, 1].

    Args:
        file_path: Scalar string tensor containing the absolute file path.
        label:     One-hot encoded label tensor.
        img_height: Target image height in pixels.
        img_width:  Target image width in pixels.

    Returns:
        Tuple of (image_tensor [H×W×3, float32], label_tensor).
    """
    raw = tf.io.read_file(file_path)
    # decode_image handles JPEG, PNG, and BMP; expand_animations=False
    # ensures we always get a 3D tensor (H×W×C).
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# ---------------------------------------------------------------------------
# Core: Augmentation layer (applied only to training data)
# ---------------------------------------------------------------------------

def build_augmentation_layer(cfg: dict) -> tf.keras.Sequential:
    """Construct a Keras Sequential augmentation sub-model.

    Augmentations are applied stochastically at training time and are
    compiled into the graph for efficiency. At inference time this layer
    is bypassed (training=False in model.predict / model.evaluate).

    Args:
        cfg: The full project configuration dictionary.

    Returns:
        A tf.keras.Sequential model consisting of augmentation layers.
    """
    aug_cfg = cfg["augmentation"]
    aug_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(aug_cfg["rotation_factor"]),
        tf.keras.layers.RandomZoom(aug_cfg["zoom_factor"]),
        tf.keras.layers.RandomBrightness(aug_cfg["brightness_delta"]),
        tf.keras.layers.RandomContrast(
            (1.0 - aug_cfg["contrast_lower"],
             aug_cfg["contrast_upper"] - 1.0)
        ),
    ], name="augmentation_pipeline")
    return aug_layer


# ---------------------------------------------------------------------------
# Core: Build file-path and label lists from directory structure
# ---------------------------------------------------------------------------

def _collect_file_paths_and_labels(
    root_dir: str,
    class_names: list
) -> tuple:
    """Walk a directory tree and collect image paths with integer labels.

    Assumes the following structure:
        root_dir/
            class_name_0/  *.jpg  *.png  ...
            class_name_1/
            ...

    Args:
        root_dir:    Path to the root directory containing class sub-folders.
        class_names: Ordered list of class names (defines label integer mapping).

    Returns:
        Tuple of (file_paths: list[str], labels: list[int]).

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no images are found.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{root_dir}'. "
            "Please download the Intel Image Classification dataset and place it "
            "under the 'data/' directory. See README.md for instructions."
        )

    file_paths, labels = [], []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  [WARNING] Class directory not found: '{class_dir}'. Skipping.")
            continue
        for fname in os.listdir(class_dir):
            if Path(fname).suffix.lower() in valid_extensions:
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)

    if len(file_paths) == 0:
        raise ValueError(
            f"No images found in '{root_dir}'. "
            "Verify the dataset structure matches the one described in README.md."
        )

    return file_paths, labels


# ---------------------------------------------------------------------------
# Core: Create tf.data.Dataset from lists
# ---------------------------------------------------------------------------

def _make_tf_dataset(
    file_paths: list,
    labels: list,
    num_classes: int,
    img_height: int,
    img_width: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    augmentation_layer=None,
    cache: bool = False
) -> tf.data.Dataset:
    """Assemble a tf.data.Dataset from file path and label lists.

    Args:
        file_paths:          List of absolute image file paths.
        labels:              List of integer class labels.
        num_classes:         Total number of classes (for one-hot encoding).
        img_height:          Target image height.
        img_width:           Target image width.
        batch_size:          Number of samples per batch.
        shuffle:             Whether to shuffle the dataset.
        seed:                Random seed for shuffling.
        augmentation_layer:  Optional Keras Sequential augmentation model.
        cache:               Whether to cache the dataset in memory after first epoch.
                             Useful for small datasets; disable if RAM is limited.

    Returns:
        A batched, prefetched tf.data.Dataset.
    """
    n = len(file_paths)

    # One-hot encode labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, one_hot_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)

    # Parallel image loading: use tf.data.AUTOTUNE for optimal CPU core usage
    ds = ds.map(
        lambda fp, lbl: _load_and_preprocess_image(fp, lbl, img_height, img_width),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)

    # Apply augmentation after batching (more efficient than per-sample)
    if augmentation_layer is not None:
        ds = ds.map(
            lambda imgs, lbls: (augmentation_layer(imgs, training=True), lbls),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Public API: build_datasets
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict) -> tuple:
    """Construct training, validation, and test tf.data datasets.

    The training directory is split 80/20 into train and validation sets
    using a deterministic stratified shuffle (to preserve class balance).
    The test directory is loaded in full as the held-out evaluation set.

    Args:
        cfg: Project configuration dictionary loaded from config.yaml.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names):
            - train_ds:    Augmented, shuffled training dataset.
            - val_ds:      Non-augmented validation dataset.
            - test_ds:     Non-augmented test dataset.
            - class_names: List of class name strings.

    Example:
        >>> cfg = load_config()
        >>> train_ds, val_ds, test_ds, class_names = build_datasets(cfg)
    """
    seed        = cfg["seed"]
    class_names = cfg["data"]["class_names"]
    num_classes = cfg["data"]["num_classes"]
    img_h       = cfg["image"]["height"]
    img_w       = cfg["image"]["width"]
    batch_size  = cfg["training"]["batch_size"]

    set_global_seed(seed)

    # -----------------------------------------------------------------------
    # Training + Validation split
    # -----------------------------------------------------------------------
    print("[DataLoader] Scanning training directory...")
    train_paths, train_labels = _collect_file_paths_and_labels(
        cfg["data"]["train_dir"], class_names
    )
    print(f"  Found {len(train_paths)} training images across {num_classes} classes.")

    # Stratified 80/20 split
    from sklearn.model_selection import train_test_split
    tr_paths, vl_paths, tr_labels, vl_labels = train_test_split(
        train_paths, train_labels,
        test_size=0.20,
        random_state=seed,
        stratify=train_labels
    )
    print(f"  Train split: {len(tr_paths)} | Validation split: {len(vl_paths)}")

    augmentation_layer = build_augmentation_layer(cfg)

    train_ds = _make_tf_dataset(
        tr_paths, tr_labels, num_classes, img_h, img_w,
        batch_size, shuffle=True, seed=seed,
        augmentation_layer=augmentation_layer,
        cache=False   # Set True if your machine has ≥16 GB RAM
    )

    val_ds = _make_tf_dataset(
        vl_paths, vl_labels, num_classes, img_h, img_w,
        batch_size, shuffle=False, seed=seed,
        augmentation_layer=None,
        cache=False
    )

    # -----------------------------------------------------------------------
    # Test set
    # -----------------------------------------------------------------------
    print("[DataLoader] Scanning test directory...")
    test_paths, test_labels = _collect_file_paths_and_labels(
        cfg["data"]["test_dir"], class_names
    )
    print(f"  Found {len(test_paths)} test images.")

    test_ds = _make_tf_dataset(
        test_paths, test_labels, num_classes, img_h, img_w,
        batch_size, shuffle=False, seed=seed,
        augmentation_layer=None,
        cache=False
    )

    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# Public API: get_sample_images (for visualization)
# ---------------------------------------------------------------------------

def get_sample_images(
    directory: str,
    class_names: list,
    n_per_class: int = 3,
    img_height: int = 128,
    img_width: int = 128,
    seed: int = 42
) -> tuple:
    """Retrieve a small sample of images and labels for visualization.

    Loads `n_per_class` images from each class directory without batching
    or augmentation — suitable for grid display in the analysis notebook.

    Args:
        directory:   Root directory with class sub-folders.
        class_names: List of class name strings.
        n_per_class: How many images to sample per class.
        img_height:  Resize height.
        img_width:   Resize width.
        seed:        Random seed for reproducible sampling.

    Returns:
        Tuple of (images: np.ndarray [N, H, W, 3], labels: list[str]).
    """
    random.seed(seed)
    images, labels_out = [], []

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir)
                 if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        sampled = random.sample(files, min(n_per_class, len(files)))
        for fname in sampled:
            raw = tf.io.read_file(os.path.join(class_dir, fname))
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.image.resize(img, [img_height, img_width])
            img = tf.cast(img, tf.float32) / 255.0
            images.append(img.numpy())
            labels_out.append(class_name)

    return np.array(images), labels_out
