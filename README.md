# Scene Recognition via Residual Convolutional Architectures: A Study on Natural Environment Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

This repository presents a systematic investigation into the application of residual convolutional neural network architectures for multi-class natural scene recognition. Leveraging the Intel Image Classification dataset — comprising approximately 25,000 high-resolution RGB images distributed across six semantically distinct environmental categories — we design, train, and evaluate a custom ResNet-inspired model augmented with stochastic regularization, adaptive learning rate scheduling, and test-time interpretability via Gradient-weighted Class Activation Mapping (Grad-CAM). The study addresses key challenges in scene understanding, namely intra-class visual variability and inter-class boundary ambiguity (e.g., glacier vs. mountain, sea vs. glacier), and reports per-class precision, recall, and F1-score alongside macro-averaged metrics. Qualitative analysis of convolutional feature maps and Grad-CAM saliency overlays provides mechanistic insight into learned discriminative representations.

---

## Table of Contents

1. [Motivation and Problem Statement](#motivation-and-problem-statement)
2. [Dataset Description](#dataset-description)
3. [Repository Structure](#repository-structure)
4. [Methodology](#methodology)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Model Architecture](#model-architecture)
   - [Training Protocol](#training-protocol)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Interpretability Analysis](#interpretability-analysis)
9. [Reproducibility](#reproducibility)
10. [Future Directions](#future-directions)
11. [Citation](#citation)
12. [License](#license)

---

## Motivation and Problem Statement

Automated scene understanding is a foundational problem in computer vision with direct implications for autonomous navigation, geographic information systems, remote sensing, and environmental monitoring. Unlike object-level recognition — where the discriminative signal is often localized to a salient foreground entity — scene-level classification requires a model to integrate holistic spatial context, texture statistics, and semantic co-occurrence patterns across the entire image plane.

Convolutional Neural Networks (CNNs) have established state-of-the-art performance on scene benchmarks; however, training such models from scratch on moderate-scale datasets requires careful architectural and regularization choices to avoid overfitting while preserving representational capacity. This project specifically investigates:

- The efficacy of residual skip connections in a shallow-to-medium depth CNN trained from scratch on ≈25K images.
- The role of stochastic data augmentation as an implicit regularizer under limited data regimes.
- The interpretability of learned feature representations via saliency-based visualization techniques.

---

## Dataset Description

**Intel Image Classification Dataset**
- **Source**: Originally released by Intel on [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Size**: ~25,000 images (150×150 pixels, RGB)
- **Split**: ~14,000 training / ~3,000 validation / ~7,000 test (prediction set)
- **Classes**: 6 semantic scene categories

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Buildings | Urban architectural structures |
| 1 | Forest | Dense woodland and vegetation |
| 2 | Glacier | Ice and snow-covered terrain |
| 3 | Mountain | Rocky elevated landforms |
| 4 | Sea | Open water bodies |
| 5 | Street | Road and urban thoroughfare views |

**Key challenges**: Glacier vs. mountain ambiguity (shared snow texture), sea vs. glacier ambiguity (shared blue-white palette), and significant intra-class illumination variance across buildings and streets.

### Downloading the Dataset

```bash
# Option 1: Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d data/

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
# Download and extract to data/ directory
```

Expected directory structure after extraction:
```
data/
├── seg_train/
│   └── seg_train/
│       ├── buildings/    (~2191 images)
│       ├── forest/       (~2271 images)
│       ├── glacier/      (~2404 images)
│       ├── mountain/     (~2512 images)
│       ├── sea/          (~2274 images)
│       └── street/       (~2382 images)
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       └── ...
└── seg_pred/
    └── seg_pred/
        └── (unlabeled images for inference)
```

---

## Repository Structure

```
intel-scene-classification/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths (single source of truth)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset ingestion, augmentation, tf.data pipeline
│   ├── model.py                 # Residual CNN architecture definition
│   ├── train.py                 # Training loop with callbacks and checkpointing
│   ├── evaluate.py              # Metrics, confusion matrix, per-class report
│   └── gradcam.py               # Grad-CAM saliency visualization
│
├── notebooks/
│   └── analysis.ipynb           # End-to-end walkthrough with inline visualizations
│
├── scripts/
│   └── download_data.py         # Automated Kaggle dataset download helper
│
├── results/                     # Auto-populated: saved models, plots, metrics
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Methodology

### Preprocessing Pipeline

All images are resized to **128×128 pixels** prior to network ingestion. Pixel intensities are normalized to `[0, 1]` via division by 255. The training split undergoes the following stochastic augmentation sequence at runtime (applied per-batch via `tf.data`):

- **Random horizontal flip** (p = 0.5)
- **Random rotation** (±15°)
- **Random zoom** (±10%)
- **Random brightness jitter** (delta = 0.1)
- **Random contrast jitter** (factor = [0.8, 1.2])

No augmentation is applied at validation or test time (standard practice for unbiased evaluation).

### Model Architecture

We design a **custom residual CNN** — termed **SceneResNet** — inspired by the residual learning framework of He et al. (2016), but significantly shallower and parameter-efficient for training from scratch on ~14K samples. The architecture proceeds as follows:

```
Input (128×128×3)
    │
    ▼
Stem Block: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
    │
    ▼
Residual Block 1: [Conv2D(64, 3×3) → BN → ReLU → Conv2D(64, 3×3) → BN] + Skip(1×1 Conv)
    │ MaxPool + Dropout(0.3)
    ▼
Residual Block 2: [Conv2D(128, 3×3) → BN → ReLU → Conv2D(128, 3×3) → BN] + Skip(1×1 Conv)
    │ MaxPool + Dropout(0.3)
    ▼
Residual Block 3: [Conv2D(256, 3×3) → BN → ReLU → Conv2D(256, 3×3) → BN] + Skip(1×1 Conv)
    │ GlobalAveragePooling
    ▼
Dense(256) → BN → ReLU → Dropout(0.5)
    │
    ▼
Dense(6) → Softmax
```

Key design rationale:
- **Residual connections** mitigate vanishing gradient issues and allow stable optimization at moderate depth.
- **Batch Normalization** after each convolution accelerates convergence and provides implicit regularization.
- **Global Average Pooling** replaces a large flatten + dense block, substantially reducing parameter count and overfitting risk.
- **Progressive channel doubling** (32 → 64 → 128 → 256) follows standard feature hierarchy conventions.

### Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial Learning Rate | 1e-3 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 50 (early stopping: patience=10) |
| Input Resolution | 128×128 |
| Random Seed | 42 |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested), Linux, macOS
- GPU recommended (NVIDIA with CUDA 11.x; CPU-only training is supported but slower)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/intel-scene-classification.git
cd intel-scene-classification

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (see Dataset section above)
```

---

## Usage

All hyperparameters are controlled from `configs/config.yaml`. Edit that file before running any script.

### Training

```bash
python src/train.py
```

Checkpoints, training curves, and logs are saved automatically to `results/`.

### Evaluation

```bash
python src/evaluate.py
```

Outputs classification report (precision, recall, F1 per class), macro/weighted averages, and a confusion matrix heatmap saved to `results/`.

### Grad-CAM Visualization

```bash
python src/gradcam.py
```

Generates saliency overlay images for a random sample from each class, saved to `results/gradcam/`.

### Jupyter Notebook (End-to-End)

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Results

> Results will be populated after training completes on your machine. Below is the expected output format.

### Quantitative Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~89–91% (expected) |
| Macro F1-Score | ~0.89 |
| Weighted F1-Score | ~0.90 |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Buildings | — | — | — |
| Forest | — | — | — |
| Glacier | — | — | — |
| Mountain | — | — | — |
| Sea | — | — | — |
| Street | — | — | — |

*Populate after running `src/evaluate.py`.*

---

## Interpretability Analysis

Grad-CAM (Selvaraju et al., 2017) generates class-discriminative localization maps by computing the gradient of the target class score with respect to the final convolutional feature maps, then weighting those feature maps by their mean gradient magnitudes. The resulting heatmap highlights image regions that most strongly influence the model's prediction.

Qualitatively, we expect the model to attend to:
- **Sky–horizon boundary** for sea and glacier discrimination.
- **Vertical structural elements** (building facades, street lamps) for buildings and street.
- **Canopy texture** for forest (high-frequency, spatially uniform activation).

---

## Reproducibility

All stochastic elements are seeded via the `seed` field in `configs/config.yaml` (default: 42). This controls:
- NumPy random state
- Python `random` module
- TensorFlow global and operation-level seeds

To reproduce results exactly, ensure the same TensorFlow and CUDA version are used, as floating-point non-determinism across hardware and driver versions can introduce minor variance.

---

## Future Directions

- **Transfer learning baseline**: Fine-tuning EfficientNet-B0 or MobileNetV2 pretrained on ImageNet-1K to benchmark the gap between training-from-scratch and pretrained representations.
- **Self-supervised pretraining**: Investigating SimCLR or BYOL on the unlabeled `seg_pred` split as a contrastive pretraining stage.
- **Knowledge distillation**: Compressing the trained SceneResNet into a smaller student network for edge deployment.
- **Mixup / CutMix augmentation**: Exploring manifold-interpolating augmentation strategies for ambiguous boundary classes (glacier/mountain).

---

## Citation

If you use this codebase or methodology in your own work, please cite:

```bibtex
@misc{yourname2025sceneresnet,
  author       = {Your Name},
  title        = {Scene Recognition via Residual Convolutional Architectures},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/<your-username>/intel-scene-classification}}
}
```

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training. *ICML 2015*.
4. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR 15(1)*.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
