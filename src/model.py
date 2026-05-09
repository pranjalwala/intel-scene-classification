"""
src/model.py
============
Definition of the SceneResNet architecture: a custom residual convolutional
neural network designed for training from scratch on moderate-scale scene
classification datasets (~14,000 images).

Architecture overview
---------------------
The network is inspired by the residual learning framework of He et al. (2016)
but is substantially shallower and parameter-efficient:

  Stem      → Conv2D(32) → BN → ReLU → MaxPool
  ResBlock1 → [Conv2D(64)×2 + 1×1 skip] → MaxPool → Dropout
  ResBlock2 → [Conv2D(128)×2 + 1×1 skip] → MaxPool → Dropout
  ResBlock3 → [Conv2D(256)×2 + 1×1 skip] → GlobalAvgPool
  Head      → Dense(256) → BN → ReLU → Dropout → Dense(6) → Softmax

Key design decisions
--------------------
- Residual skip connections: Alleviate vanishing gradients and allow
  meaningful gradient flow even at moderate depth. Each skip projects the
  input channels to match the residual branch output via a 1×1 convolution
  (projection shortcut).
- Batch Normalisation: Applied after every Conv2D, before the non-linearity,
  following the pre-activation convention of He et al. (2016) Identity
  Mappings paper (though here applied in the standard post-activation
  position for simplicity).
- Global Average Pooling: Replaces a large Flatten → Dense block,
  dramatically reducing parameter count (from ~4M to ~500K in the head)
  and improving generalisation under limited data.
- Progressive channel doubling: 32 → 64 → 128 → 256 follows the standard
  feature hierarchy convention, balancing spatial resolution reduction with
  increased representational capacity.

Reference
---------
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
image recognition. CVPR 2016.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input


# ---------------------------------------------------------------------------
# Building block: Residual Block
# ---------------------------------------------------------------------------

def residual_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """A single residual block with a projection shortcut.

    Structure:
        ┌────────────────────────────────────────────┐
        │  Conv2D(filters, 3×3) → BN → ReLU          │
        │  Conv2D(filters, 3×3) → BN                 │  (main branch)
        └──────────────────────┬─────────────────────┘
                               │
        Conv2D(filters, 1×1) → BN               (projection shortcut)
                               │
                           Add → ReLU

    The projection shortcut (1×1 conv) is always applied to align channel
    dimensions between input and output, regardless of whether the channel
    count changes. This is mathematically equivalent to the "option B"
    shortcut in He et al. (2016).

    Args:
        x:       Input feature map tensor.
        filters: Number of convolutional filters for the main branch.
        name:    String prefix for all sub-layer names (for interpretability).

    Returns:
        Output feature map tensor with shape (H, W, filters).
    """
    # --- Main branch ---
    fx = layers.Conv2D(
        filters, kernel_size=3, padding="same",
        use_bias=False, name=f"{name}_conv1"
    )(x)
    fx = layers.BatchNormalization(name=f"{name}_bn1")(fx)
    fx = layers.ReLU(name=f"{name}_relu1")(fx)

    fx = layers.Conv2D(
        filters, kernel_size=3, padding="same",
        use_bias=False, name=f"{name}_conv2"
    )(fx)
    fx = layers.BatchNormalization(name=f"{name}_bn2")(fx)

    # --- Projection shortcut (1×1 conv to match channels) ---
    shortcut = layers.Conv2D(
        filters, kernel_size=1, padding="same",
        use_bias=False, name=f"{name}_proj"
    )(x)
    shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    # --- Residual addition + activation ---
    out = layers.Add(name=f"{name}_add")([fx, shortcut])
    out = layers.ReLU(name=f"{name}_relu2")(out)
    return out


# ---------------------------------------------------------------------------
# Public API: build_model
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> tf.keras.Model:
    """Construct and return the SceneResNet model.

    The model is built using the Keras Functional API (rather than Sequential)
    to make the residual connections explicit in the computation graph and to
    allow straightforward layer lookup for Grad-CAM.

    Args:
        cfg: Project configuration dictionary loaded from config.yaml.

    Returns:
        An uncompiled tf.keras.Model instance.

    Example:
        >>> cfg = load_config("configs/config.yaml")
        >>> model = build_model(cfg)
        >>> model.summary()
    """
    img_h       = cfg["image"]["height"]
    img_w       = cfg["image"]["width"]
    img_c       = cfg["image"]["channels"]
    num_classes = cfg["data"]["num_classes"]

    stem_filters      = cfg["model"]["stem_filters"]
    res_filters       = cfg["model"]["residual_filters"]   # [64, 128, 256]
    drop_residual     = cfg["model"]["dropout_residual"]
    dense_units       = cfg["model"]["dense_units"]
    drop_dense        = cfg["model"]["dropout_dense"]

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------
    inputs = Input(shape=(img_h, img_w, img_c), name="input_image")

    # -----------------------------------------------------------------------
    # Stem Block
    # Purpose: Rapidly reduce spatial resolution while extracting low-level
    # edge and texture features.
    # -----------------------------------------------------------------------
    x = layers.Conv2D(
        stem_filters, kernel_size=3, padding="same",
        use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="stem_pool")(x)
    # Shape: (64, 64, 32)

    # -----------------------------------------------------------------------
    # Residual Block 1 — 64 filters
    # -----------------------------------------------------------------------
    x = residual_block(x, filters=res_filters[0], name="res1")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="res1_pool")(x)
    x = layers.Dropout(drop_residual, name="res1_dropout")(x)
    # Shape: (32, 32, 64)

    # -----------------------------------------------------------------------
    # Residual Block 2 — 128 filters
    # -----------------------------------------------------------------------
    x = residual_block(x, filters=res_filters[1], name="res2")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="res2_pool")(x)
    x = layers.Dropout(drop_residual, name="res2_dropout")(x)
    # Shape: (16, 16, 128)

    # -----------------------------------------------------------------------
    # Residual Block 3 — 256 filters
    # This is the final convolutional block; its feature maps are used for
    # Grad-CAM visualisation (see src/gradcam.py).
    # -----------------------------------------------------------------------
    x = residual_block(x, filters=res_filters[2], name="res3")
    # Shape: (16, 16, 256)  — note: no MaxPool here to preserve spatial detail
    # for Grad-CAM

    # -----------------------------------------------------------------------
    # Global Average Pooling
    # Aggregates each feature map into a single scalar, replacing a large
    # Flatten → Dense block and substantially reducing overfitting risk.
    # -----------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # Shape: (256,)

    # -----------------------------------------------------------------------
    # Classification Head
    # -----------------------------------------------------------------------
    x = layers.Dense(dense_units, use_bias=False, name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(name="head_relu")(x)
    x = layers.Dropout(drop_dense, name="head_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SceneResNet")
    return model


# ---------------------------------------------------------------------------
# Public API: compile_model
# ---------------------------------------------------------------------------

def compile_model(model: tf.keras.Model, cfg: dict) -> tf.keras.Model:
    """Compile the model with the optimizer and loss specified in config.

    Args:
        model: An uncompiled tf.keras.Model.
        cfg:   Project configuration dictionary.

    Returns:
        The compiled model (same object, modified in-place and returned
        for chaining convenience).
    """
    lr = cfg["training"]["initial_lr"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ]
    )
    return model
