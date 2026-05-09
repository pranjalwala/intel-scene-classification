# Intel Scene Classification using Residual CNNs

A research-oriented deep learning pipeline for multi-class natural scene classification using a custom ResNet-style convolutional neural network implemented in TensorFlow/Keras.

---

## Overview

This project focuses on large-scale image scene classification using the Intel Image Classification dataset containing approximately 25,000 real-world RGB images across six semantic scene categories:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

Unlike basic classroom CNN projects built on CIFAR-10 or MNIST, this repository implements a modular and reproducible computer vision workflow inspired by practical deep learning research pipelines.

The project includes:

* Custom residual CNN architecture
* TensorFlow `tf.data` input pipeline
* Data augmentation and preprocessing
* Training checkpointing and callbacks
* Evaluation metrics and confusion matrix analysis
* Grad-CAM explainability visualizations
* Config-driven experimentation
* Modular source code organization

---

## Project Objectives

The primary goals of this project are:

* Build a scalable and reproducible image classification pipeline
* Implement residual learning instead of a basic sequential CNN
* Analyze model performance beyond accuracy using F1-score and confusion matrices
* Explore interpretability using Grad-CAM saliency visualization
* Structure the repository similar to a research or production ML workflow

---

## Dataset

### Intel Image Classification Dataset

Dataset Source:

[https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

### Dataset Characteristics

| Property     | Value                                 |
| ------------ | ------------------------------------- |
| Total Images | ~25,000                               |
| Classes      | 6                                     |
| Image Type   | RGB Natural Scenes                    |
| Resolution   | Variable                              |
| Domain       | Computer Vision / Scene Understanding |

### Scene Categories

| Class     | Description                           |
| --------- | ------------------------------------- |
| Buildings | Urban infrastructure and architecture |
| Forest    | Dense vegetation and woodland scenes  |
| Glacier   | Snow and ice landscapes               |
| Mountain  | Rocky mountain environments           |
| Sea       | Ocean and coastal scenes              |
| Street    | Urban road and city environments      |

---

## Model Architecture

The project implements a custom residual convolutional neural network inspired by ResNet architectures.

### Key Architectural Features

* Residual skip connections
* Projection shortcuts using 1×1 convolutions
* Batch normalization
* Dropout regularization
* Global average pooling
* Configurable filter depth
* Softmax multi-class output layer

### Why Residual Learning?

Residual connections help mitigate:

* Vanishing gradients
* Optimization degradation in deeper networks
* Poor convergence stability

This enables more stable training compared to traditional sequential CNN architectures.

---

## Training Pipeline

### Input Pipeline

The project uses TensorFlow `tf.data` for efficient GPU-compatible data loading.

Features include:

* Stratified train-validation splitting
* Dataset shuffling
* Parallel loading
* Prefetching with `AUTOTUNE`
* Batch processing

### Data Augmentation

Training augmentation includes:

* Horizontal flipping
* Random rotation
* Zoom augmentation
* Brightness adjustment
* Contrast variation

These techniques improve model generalization and robustness.

---

## Evaluation Methodology

Model evaluation goes beyond raw accuracy.

### Metrics Used

* Accuracy
* Precision
* Recall
* F1-score
* Macro-average F1
* Weighted-average F1
* Confusion Matrix

### Why F1-score?

F1-score provides a balanced view of:

* False positives
* False negatives
* Class-wise prediction quality

This is especially important for multi-class classification problems.

---

## Grad-CAM Explainability

The repository includes Grad-CAM visualization for interpretability analysis.

Grad-CAM highlights spatial image regions contributing most strongly to model predictions.

This helps analyze:

* Attention localization
* Feature learning behavior
* Prediction explainability
* Failure cases

---

# Results

## Training Curves

![Training Curves](assets/training_curves.png)

---

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

## Per-Class F1 Scores

![Per Class F1](assets/per_class_f1.png)

---

## Grad-CAM Visualizations

### Buildings

![Buildings GradCAM](assets/buildings_gradcam.png)

### Forest

![Forest GradCAM](assets/forest_gradcam.png)

### Glacier

![Glacier GradCAM](assets/glacier_gradcam.png)

### Mountain

![Mountain GradCAM](assets/mountain_gradcam.png)

### Sea

![Sea GradCAM](assets/sea_gradcam.png)

### Street

![Street GradCAM](assets/street_gradcam.png)

---

## Final Evaluation Metrics

| Metric              | Score  |
| ------------------- | ------ |
| Test Accuracy       | 15.23% |
| Macro Average F1    | 0.0579 |
| Weighted Average F1 | 0.0564 |

> Note: The above metrics correspond to an early-stage verification run used to validate the full training pipeline. Longer GPU-based training is expected to produce significantly higher classification performance.

---

## Repository Structure

```text
intel-scene-classification/
│
├── assets/                     # README visualizations
├── configs/
│   └── config.yaml             # Centralized hyperparameters
│
├── notebooks/
│   └── analysis.ipynb          # Interactive experimentation notebook
│
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   ├── plots/
│   └── gradcam/
│
├── scripts/
│   └── download_data.py
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/pranjalwala/intel-scene-classification.git
cd intel-scene-classification
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### Kaggle Dataset Download

```bash
kaggle datasets download -d puneet6060/intel-image-classification -p data
```

### Extract Dataset

```bash
unzip intel-image-classification.zip -d data
```

Expected structure:

```text
data/
├── seg_train/
├── seg_test/
└── seg_pred/
```

---

## Training

Run training:

```bash
python src/train.py
```

The pipeline automatically handles:

* model checkpointing
* learning rate scheduling
* early stopping
* training history logging

---

## Evaluation

Run evaluation:

```bash
python src/evaluate.py
```

Generated outputs:

* classification report
* confusion matrix
* F1-score plots
* JSON metric summary

---

## Grad-CAM Visualization

Generate explainability visualizations:

```bash
python src/gradcam.py
```

Generated outputs are stored in:

```text
results/gradcam/
```

---

## Technologies Used

| Category        | Tools               |
| --------------- | ------------------- |
| Deep Learning   | TensorFlow, Keras   |
| Data Processing | NumPy, Pandas       |
| Visualization   | Matplotlib, Seaborn |
| ML Utilities    | scikit-learn        |
| Experimentation | Jupyter Notebook    |
| Version Control | Git, GitHub         |

---

## Future Improvements

Potential future extensions include:

* Transfer learning with EfficientNet or ResNet50
* TensorBoard integration
* Streamlit deployment interface
* Hyperparameter optimization
* Mixed precision GPU training
* Advanced augmentation policies
* Experiment tracking with Weights & Biases

---

## Research Relevance

This repository demonstrates concepts relevant to:

* Computer Vision
* Scene Understanding
* Deep Convolutional Networks
* Residual Learning
* Explainable AI (XAI)
* Reproducible Machine Learning

---

## Author

Pranjal Wala

GitHub:

[https://github.com/pranjalwala](https://github.com/pranjalwala)

---

## License

This project is released under the MIT License.
