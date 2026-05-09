"""
src/gradcam.py
==============
Gradient-weighted Class Activation Mapping (Grad-CAM) for SceneResNet.

Background
----------
Grad-CAM (Selvaraju et al., ICCV 2017) produces class-discriminative
localisation maps without architectural modification. Given an input image
and a target class c, it:

  1. Performs a forward pass and identifies the final convolutional feature
     maps A^k (of the last residual block: `res3_relu2`).
  2. Backpropagates the gradient of the class score y^c w.r.t. each A^k.
  3. Computes importance weights α^c_k = global_average_pool(∂y^c / ∂A^k).
  4. Produces the heatmap L^c = ReLU(Σ_k α^c_k · A^k).
  5. Upsamples L^c to input resolution and overlays it on the original image.

The ReLU in step 4 retains only features with positive influence on the
target class (i.e., features that push the score up).

Reference
---------
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-CAM: Visual explanations from deep networks via
gradient-based localization. ICCV 2017.

Usage
-----
    python src/gradcam.py

Saves one composite figure per class to results/gradcam/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from src.data_loader import load_config, set_global_seed, get_sample_images


# ---------------------------------------------------------------------------
# Core: Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str = "res3_relu2"
) -> np.ndarray:
    """Compute the Grad-CAM heatmap for a single image and target class.

    Args:
        model:                The trained SceneResNet model.
        image:                A single image as a float32 NumPy array in
                              [0, 1], shape (H, W, 3).
        class_idx:            Integer index of the target class for which
                              the Grad-CAM is computed.
        last_conv_layer_name: Name of the final convolutional activation
                              layer in the model graph. For SceneResNet
                              this is 'res3_relu2'.

    Returns:
        heatmap: Float32 NumPy array of shape (H, W), values in [0, 1],
                 representing the class-discriminative saliency map at
                 input resolution.
    """
    # Build a sub-model that outputs both the final conv layer activations
    # and the model's final class predictions simultaneously.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    img_tensor = tf.cast(tf.expand_dims(image, axis=0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Extract the scalar score for the target class
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. the last conv feature maps
    grads = tape.gradient(loss, conv_outputs)  # shape: (1, H', W', C)

    # Global average pooling of gradients → importance weights α^c_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]              # shape: (H', W', C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]   # (H', W', 1)
    heatmap = tf.squeeze(heatmap)               # (H', W')

    # ReLU: keep only positive contributions
    heatmap = tf.nn.relu(heatmap).numpy()

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation: Overlay heatmap on image
# ---------------------------------------------------------------------------

def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> np.ndarray:
    """Superimpose a Grad-CAM heatmap onto the original image.

    Resizes the heatmap to match the input image resolution, applies a
    colourmap, and blends with the original image.

    Args:
        image:    Original image, float32, shape (H, W, 3), values in [0, 1].
        heatmap:  Grad-CAM heatmap, float32, shape (H', W'), values in [0, 1].
        alpha:    Blending coefficient for the heatmap overlay (0 = no overlay,
                  1 = heatmap only).
        colormap: Matplotlib colourmap name for the heatmap.

    Returns:
        Blended image as float32 NumPy array, shape (H, W, 3), values in [0, 1].
    """
    h, w = image.shape[:2]

    # Resize heatmap to input resolution using bilinear interpolation
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [h, w], method="bilinear"
    ).numpy().squeeze()

    # Apply colourmap (returns RGBA; take only RGB)
    cmap      = plt.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap_resized)[..., :3]  # (H, W, 3), float64

    overlay = (1 - alpha) * image + alpha * heatmap_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# ---------------------------------------------------------------------------
# Visualisation: Per-class grid of Grad-CAM overlays
# ---------------------------------------------------------------------------

def plot_gradcam_grid(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: list,
    class_names: list,
    output_dir: str,
    last_conv_layer_name: str = "res3_relu2",
    n_per_class: int = 3
) -> None:
    """Generate one Grad-CAM composite figure per class.

    For each class, creates an n_per_class × 2 grid showing:
        Column 1: Original image
        Column 2: Grad-CAM overlay

    Saves each figure as results/gradcam/<class_name>_gradcam.png.

    Args:
        model:                Trained SceneResNet model.
        images:               NumPy array of images, shape (N, H, W, 3).
        labels:               List of class name strings corresponding to images.
        class_names:          All class names (to iterate over).
        output_dir:           Directory to save output figures.
        last_conv_layer_name: Target convolutional layer for gradient extraction.
        n_per_class:          Number of examples to show per class.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in class_names:
        class_idx = class_names.index(class_name)

        # Select images of this class
        class_images = [img for img, lbl in zip(images, labels) if lbl == class_name]
        if len(class_images) == 0:
            print(f"  [WARNING] No samples found for class '{class_name}'. Skipping.")
            continue

        n = min(n_per_class, len(class_images))
        fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
        if n == 1:
            axes = np.expand_dims(axes, 0)

        fig.suptitle(
            f"Grad-CAM Visualisation — Class: '{class_name.capitalize()}'",
            fontsize=13, fontweight="bold"
        )

        for row_idx in range(n):
            img = class_images[row_idx]

            heatmap = compute_gradcam(model, img, class_idx, last_conv_layer_name)
            overlay = overlay_gradcam(img, heatmap, alpha=0.45)

            # Original
            ax = axes[row_idx, 0]
            ax.imshow(img)
            ax.set_title("Original", fontsize=9)
            ax.axis("off")

            # Grad-CAM overlay
            ax = axes[row_idx, 1]
            ax.imshow(overlay)
            ax.set_title("Grad-CAM Overlay", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{class_name}_gradcam.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  SceneResNet — Grad-CAM Visualisation")
    print("=" * 60)

    cfg = load_config("configs/config.yaml")
    set_global_seed(cfg["seed"])

    class_names    = cfg["data"]["class_names"]
    best_model_path = cfg["output"]["best_model_path"]
    gradcam_dir     = cfg["output"]["gradcam_dir"]

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    if not os.path.exists(best_model_path):
        print(f"\n[ERROR] No checkpoint found at '{best_model_path}'.")
        print("  Please run `python src/train.py` first.")
        sys.exit(1)

    print(f"\n[GradCAM] Loading model from: {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)

    # Verify the target layer exists
    layer_names = [l.name for l in model.layers]
    target_layer = "res3_relu2"
    if target_layer not in layer_names:
        print(f"[ERROR] Layer '{target_layer}' not found in model.")
        print(f"  Available layers: {layer_names}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load sample images
    # -----------------------------------------------------------------------
    print("\n[GradCAM] Loading sample images from test set...")
    images, labels = get_sample_images(
        directory=cfg["data"]["test_dir"],
        class_names=class_names,
        n_per_class=3,
        img_height=cfg["image"]["height"],
        img_width=cfg["image"]["width"],
        seed=cfg["seed"]
    )
    print(f"  Loaded {len(images)} images.")

    # -----------------------------------------------------------------------
    # Generate Grad-CAM figures
    # -----------------------------------------------------------------------
    print(f"\n[GradCAM] Generating saliency maps → {gradcam_dir}/")
    plot_gradcam_grid(
        model=model,
        images=images,
        labels=labels,
        class_names=class_names,
        output_dir=gradcam_dir,
        last_conv_layer_name=target_layer,
        n_per_class=3
    )

    print("\n[GradCAM] All visualisations complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
