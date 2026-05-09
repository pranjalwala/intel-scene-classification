"""
src/train.py
============
Training entry point for the SceneResNet model.

Execution flow
--------------
1. Load configuration from configs/config.yaml.
2. Set global random seeds for reproducibility.
3. Build tf.data pipelines (train, val, test) via data_loader.
4. Construct and compile the SceneResNet model via model.py.
5. Register Keras callbacks:
     - ModelCheckpoint  : Persist the best validation-loss checkpoint.
     - EarlyStopping    : Halt training when val_loss ceases to improve.
     - ReduceLROnPlateau: Halve the learning rate on validation loss plateau.
     - CSVLogger        : Append per-epoch metrics to a CSV log file.
6. Run model.fit().
7. Serialise the training history to JSON for downstream analysis.
8. Generate and save training curve plots (accuracy and loss).

Usage
-----
From the project root directory:

    python src/train.py

All output artefacts are written to paths specified in configs/config.yaml
under the `output` key.
"""

import os
import sys
import json
import time

# Resolve imports relative to project root when called as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import load_config, set_global_seed, build_datasets
from src.model import build_model, compile_model


# ---------------------------------------------------------------------------
# Callback factories
# ---------------------------------------------------------------------------

def build_callbacks(cfg: dict) -> list:
    """Construct the list of Keras training callbacks from configuration.

    Returns:
        List of tf.keras.callbacks.Callback instances.
    """
    out_cfg  = cfg["output"]
    lr_cfg   = cfg["lr_schedule"]
    es_cfg   = cfg["early_stopping"]

    os.makedirs(out_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(out_cfg["metrics_dir"], exist_ok=True)

    # --- Model Checkpoint ---
    # Saves the model weights whenever validation loss improves.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_cfg["best_model_path"],
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # --- Early Stopping ---
    # Restores the best weights upon termination.
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        restore_best_weights=es_cfg["restore_best_weights"],
        verbose=1
    )

    # --- Learning Rate Reduction on Plateau ---
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=lr_cfg["monitor"],
        factor=lr_cfg["factor"],
        patience=lr_cfg["patience"],
        min_lr=lr_cfg["min_lr"],
        verbose=lr_cfg["verbose"]
    )

    # --- CSV Logger ---
    # Appends one row per epoch to a CSV for later analysis.
    csv_log_path = os.path.join(out_cfg["metrics_dir"], "training_log.csv")
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
        csv_log_path, separator=",", append=False
    )

    return [checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_cb]


# ---------------------------------------------------------------------------
# Visualisation: Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_dir: str) -> None:
    """Generate and save accuracy and loss training curves.

    Produces two side-by-side subplots:
        Left:  Training vs. validation accuracy across epochs.
        Right: Training vs. validation categorical cross-entropy loss.

    Args:
        history:    Dictionary of metric name → list of per-epoch values,
                    as returned by tf.keras History.history.
        output_dir: Directory path where the figure will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SceneResNet — Training History", fontsize=14, fontweight="bold")

    # --- Accuracy ---
    ax = axes[0]
    ax.plot(epochs, history["accuracy"],     label="Train Accuracy",      color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], label="Validation Accuracy",  color="#FF5722", linewidth=2, linestyle="--")
    ax.set_title("Categorical Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[1]
    ax.plot(epochs, history["loss"],     label="Train Loss",      color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss",  color="#FF5722", linewidth=2, linestyle="--")
    ax.set_title("Categorical Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Train] Training curves saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  SceneResNet — Intel Image Classification")
    print("  Training Pipeline")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Configuration and seeding
    # -----------------------------------------------------------------------
    cfg = load_config("configs/config.yaml")
    set_global_seed(cfg["seed"])
    print(f"\n[Config] Seed: {cfg['seed']}")
    print(f"[Config] Input resolution: {cfg['image']['height']}×{cfg['image']['width']}")
    print(f"[Config] Batch size: {cfg['training']['batch_size']}")
    print(f"[Config] Max epochs: {cfg['training']['epochs']}")

    # -----------------------------------------------------------------------
    # 2. Build data pipelines
    # -----------------------------------------------------------------------
    print("\n[Step 1/4] Building data pipelines...")
    train_ds, val_ds, test_ds, class_names = build_datasets(cfg)
    print(f"  Classes: {class_names}")

    # -----------------------------------------------------------------------
    # 3. Build and compile model
    # -----------------------------------------------------------------------
    print("\n[Step 2/4] Constructing SceneResNet architecture...")
    model = build_model(cfg)
    model = compile_model(model, cfg)
    model.summary(line_length=80)

    total_params = model.count_params()
    print(f"\n  Total trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # 4. Callbacks
    # -----------------------------------------------------------------------
    print("\n[Step 3/4] Registering callbacks...")
    callbacks = build_callbacks(cfg)
    print(f"  Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger")

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------
    print("\n[Step 4/4] Beginning training...")
    t_start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    elapsed = time.time() - t_start
    print(f"\n[Train] Training complete in {elapsed / 60:.1f} minutes.")

    # -----------------------------------------------------------------------
    # 6. Save training history
    # -----------------------------------------------------------------------
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    os.makedirs(os.path.dirname(cfg["output"]["history_path"]), exist_ok=True)
    with open(cfg["output"]["history_path"], "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Train] History saved to: {cfg['output']['history_path']}")

    # -----------------------------------------------------------------------
    # 7. Training curves
    # -----------------------------------------------------------------------
    plot_training_curves(history_dict, cfg["output"]["plots_dir"])

    # -----------------------------------------------------------------------
    # 8. Quick test-set evaluation
    # -----------------------------------------------------------------------
    print("\n[Train] Evaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    print("\n  Test set results:")
    for name, val in zip(metric_names, test_results):
        print(f"    {name}: {val:.4f}")

    print("\n[Train] Run `python src/evaluate.py` for full per-class metrics and confusion matrix.")
    print("[Train] Run `python src/gradcam.py` for Grad-CAM visualisations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
