# Scene Recognition via Residual Convolutional Architectures

Research-oriented scene classification pipeline using a custom ResNet-style convolutional neural network built with TensorFlow/Keras for multi-class natural scene understanding.

---

## Overview

This project investigates residual convolutional architectures for classifying natural environments using the Intel Image Classification dataset (~25,000 RGB images across six semantic categories).

The repository focuses on:
- residual CNN design
- tf.data training pipelines
- stochastic augmentation strategies
- Grad-CAM interpretability
- per-class evaluation metrics
- reproducible experimentation workflows

---

## Dataset

### Intel Image Classification Dataset

| Class ID | Category |
|---|---|
| 0 | Buildings |
| 1 | Forest |
| 2 | Glacier |
| 3 | Mountain |
| 4 | Sea |
| 5 | Street |

Dataset Source:  
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

---

## Repository Structure

```text
intel-scene-classification/
│
├── assets/
├── configs/
├── notebooks/
├── results/
├── scripts/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Methodology

### Preprocessing Pipeline

The training workflow includes:
- image resizing to 128×128
- normalization to `[0,1]`
- random horizontal flipping
- random rotation and zoom augmentation
- brightness and contrast perturbation

---

### Model Architecture

The custom **SceneResNet** architecture incorporates:

- residual skip connections
- 1×1 projection shortcuts
- batch normalization
- dropout regularization
- global average pooling
- softmax classification head

The architecture is intentionally lightweight and optimized for training from scratch on moderate-scale image datasets.

---

# Results

## Quantitative Performance

| Metric | Value |
|---|---|
| Test Accuracy | 15.23% |
| Macro F1-Score | 0.0579 |
| Weighted F1-Score | 0.0564 |

> These results correspond to an early-stage verification run used to validate the complete training and evaluation pipeline.

---

## Training Curves

<p align="center">
  <img src="assets/training_curves.png" width="850">
</p>

---

## Confusion Matrix

<p align="center">
  <img src="assets/confusion_matrix.png" width="750">
</p>

---

## Per-Class F1 Scores

<p align="center">
  <img src="assets/per_class_f1.png" width="750">
</p>

---

## Grad-CAM Visualizations

### Buildings

<p align="center">
  <img src="assets/buildings_gradcam.png" width="750">
</p>

---

### Forest

<p align="center">
  <img src="assets/forest_gradcam.png" width="750">
</p>

---

### Glacier

<p align="center">
  <img src="assets/glacier_gradcam.png" width="750">
</p>

---

### Mountain

<p align="center">
  <img src="assets/mountain_gradcam.png" width="750">
</p>

---

### Sea

<p align="center">
  <img src="assets/sea_gradcam.png" width="750">
</p>

---

### Street

<p align="center">
  <img src="assets/street_gradcam.png" width="750">
</p>

---

## Installation

```bash
git clone https://github.com/pranjalwala/intel-scene-classification.git
cd intel-scene-classification

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

### Grad-CAM Generation

```bash
python src/gradcam.py
```

---

## Technologies Used

- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Git & GitHub

---

## Future Directions

- transfer learning baselines
- hyperparameter optimization
- mixed precision GPU training
- TensorBoard integration
- lightweight deployment workflows

---

## References

1. He et al. — Deep Residual Learning for Image Recognition (CVPR 2016)
2. Selvaraju et al. — Grad-CAM (ICCV 2017)
3. Ioffe & Szegedy — Batch Normalization (ICML 2015)

---

## Author

Pranjal Wala

GitHub:  
https://github.com/pranjalwala
