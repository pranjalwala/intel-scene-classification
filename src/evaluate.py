"""
src/evaluate.py
===============
Comprehensive evaluation of the trained SceneResNet model on the held-out
test set.

Outputs produced
----------------
1. Console: Full sklearn classification report (precision, recall, F1-score
   per class + macro/weighted averages).
2. results/metrics/classification_report.txt  — plain-text report.
3. results/plots/confusion_matrix.png         — normalised heatmap.
4. results/plots/per_class_f1.png             — bar chart of per-class F1.

Prerequisites
-------------
- `python src/train.py` must have completed and saved a checkpoint to
  the path specified in configs/config.yaml → output.best_model_path.

Usage
-----
    python src/evaluate.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_loader import load_config, set_global_seed, build_datasets


# ---------------------------------------------------------------------------
# Prediction utility
# ---------------------------------------------------------------------------

def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple:
    """Run inference over a batched dataset and return flat arrays.

    Args:
        model:   A compiled and trained tf.keras.Model.
        dataset: A batched tf.data.Dataset yielding (images, one_hot_labels).

    Returns:
        Tuple of:
            y_true (np.ndarray, int): Ground-truth class indices.
            y_pred (np.ndarray, int): Predicted class indices (argmax of softmax).
            y_prob (np.ndarray, float): Full softmax probability matrix [N × C].
    """
    all_probs  = []
    all_labels = []

    for images, labels in dataset:
        probs = model(images, training=False).numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob  = np.concatenate(all_probs, axis=0)
    y_true  = np.argmax(np.concatenate(all_labels, axis=0), axis=1)
    y_pred  = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Visualisation: Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str
) -> None:
    """Plot and save a row-normalised confusion matrix as a seaborn heatmap.

    Row normalisation expresses each cell as the fraction of samples of the
    true class predicted as each class, making it easy to identify which
    classes are most commonly confused.

    Args:
        y_true:      Ground-truth integer class labels.
        y_pred:      Predicted integer class labels.
        class_names: Ordered list of class name strings.
        output_path: File path to save the figure (PNG).
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )
    ax.set_title("Normalised Confusion Matrix — SceneResNet\n(values = fraction of true class)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to: {output_path}")


# ---------------------------------------------------------------------------
# Visualisation: Per-class F1 bar chart
# ---------------------------------------------------------------------------

def plot_per_class_f1(
    report_dict: dict,
    class_names: list,
    output_path: str
) -> None:
    """Bar chart of per-class F1-scores with macro average reference line.

    Args:
        report_dict: Dictionary as returned by sklearn.classification_report
                     with output_dict=True.
        class_names: Ordered list of class name strings.
        output_path: File path to save the figure (PNG).
    """
    f1_scores = [report_dict[name]["f1-score"] for name in class_names]
    macro_f1  = report_dict["macro avg"]["f1-score"]

    colours = ["#2196F3" if f >= macro_f1 else "#FF5722" for f in f1_scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, f1_scores, color=colours, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Annotate bars with values
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.axhline(macro_f1, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Macro Avg F1 = {macro_f1:.3f}", zorder=4)

    ax.set_title("Per-Class F1-Score — SceneResNet", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Per-class F1 chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  SceneResNet — Evaluation Pipeline")
    print("=" * 60)

    cfg = load_config("configs/config.yaml")
    set_global_seed(cfg["seed"])

    class_names    = cfg["data"]["class_names"]
    best_model_path = cfg["output"]["best_model_path"]
    plots_dir       = cfg["output"]["plots_dir"]
    metrics_dir     = cfg["output"]["metrics_dir"]

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    if not os.path.exists(best_model_path):
        print(f"\n[ERROR] No checkpoint found at '{best_model_path}'.")
        print("  Please run `python src/train.py` first.")
        sys.exit(1)

    print(f"\n[Evaluate] Loading model from: {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)
    print("[Evaluate] Model loaded successfully.")

    # -----------------------------------------------------------------------
    # Build test dataset
    # -----------------------------------------------------------------------
    print("\n[Evaluate] Building test dataset pipeline...")
    _, _, test_ds, _ = build_datasets(cfg)

    # -----------------------------------------------------------------------
    # Run predictions
    # -----------------------------------------------------------------------
    print("[Evaluate] Running inference on test set (this may take a moment)...")
    y_true, y_pred, y_prob = get_predictions(model, test_ds)
    print(f"  Processed {len(y_true)} test samples.")

    # -----------------------------------------------------------------------
    # Classification report
    # -----------------------------------------------------------------------
    report_str  = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True)

    print("\n" + "─" * 60)
    print("  Classification Report")
    print("─" * 60)
    print(report_str)

    report_path = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("SceneResNet — Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    print(f"[Evaluate] Report saved to: {report_path}")

    # -----------------------------------------------------------------------
    # Overall accuracy
    # -----------------------------------------------------------------------
    overall_acc = np.mean(y_true == y_pred)
    print(f"\n  Overall Test Accuracy : {overall_acc * 100:.2f}%")
    print(f"  Macro-Avg F1-Score    : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Weighted-Avg F1-Score : {report_dict['weighted avg']['f1-score']:.4f}")

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # -----------------------------------------------------------------------
    # Per-class F1 chart
    # -----------------------------------------------------------------------
    f1_path = os.path.join(plots_dir, "per_class_f1.png")
    plot_per_class_f1(report_dict, class_names, f1_path)

    # -----------------------------------------------------------------------
    # Save numerical summary as JSON
    # -----------------------------------------------------------------------
    summary = {
        "overall_accuracy": float(overall_acc),
        "macro_avg_f1":     float(report_dict["macro avg"]["f1-score"]),
        "weighted_avg_f1":  float(report_dict["weighted avg"]["f1-score"]),
        "per_class":        {
            name: {
                "precision": float(report_dict[name]["precision"]),
                "recall":    float(report_dict[name]["recall"]),
                "f1":        float(report_dict[name]["f1-score"]),
                "support":   int(report_dict[name]["support"]),
            }
            for name in class_names
        }
    }
    summary_path = os.path.join(metrics_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Evaluate] Numeric summary saved to: {summary_path}")

    print("\n[Evaluate] Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
