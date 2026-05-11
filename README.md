\# Intel Scene Classification using Residual CNNs



Research-oriented scene classification pipeline using a custom ResNet-style CNN built with TensorFlow/Keras for multi-class natural scene understanding.



\---



\## Overview



This project implements a modular deep learning workflow for classifying natural scenes from the Intel Image Classification dataset (\~25,000 images across 6 classes).



Key features:



\- Custom ResNet-style CNN with residual connections

\- TensorFlow `tf.data` GPU pipeline

\- Data augmentation and preprocessing

\- Grad-CAM explainability

\- Confusion matrix and F1-score evaluation

\- Config-driven experimentation

\- Modular training and evaluation scripts



\---



\## Dataset



Dataset: Intel Image Classification Dataset



Classes:

\- Buildings

\- Forest

\- Glacier

\- Mountain

\- Sea

\- Street



Source:  

https://www.kaggle.com/datasets/puneet6060/intel-image-classification



\---



\## Model Architecture



The model is inspired by residual learning architectures and includes:



\- Residual skip connections

\- 1×1 projection shortcuts

\- Batch normalization

\- Dropout regularization

\- Global average pooling

\- Softmax classification head



\---



\# Results



\## Training Curves



!\[Training Curves](assets/training\_curves.png)



\---



\## Confusion Matrix



!\[Confusion Matrix](assets/confusion\_matrix.png)



\---



\## Per-Class F1 Scores



!\[Per Class F1](assets/per\_class\_f1.png)



\---



\## Grad-CAM Visualizations



| Buildings | Forest |

|---|---|

| !\[](assets/buildings\_gradcam.png) | !\[](assets/forest\_gradcam.png) |



| Glacier | Mountain |

|---|---|

| !\[](assets/glacier\_gradcam.png) | !\[](assets/mountain\_gradcam.png) |



| Sea | Street |

|---|---|

| !\[](assets/sea\_gradcam.png) | !\[](assets/street\_gradcam.png) |



\---



\## Evaluation Metrics



| Metric | Score |

|---|---|

| Test Accuracy | 15.23% |

| Macro F1 Score | 0.0579 |

| Weighted F1 Score | 0.0564 |



> Current metrics correspond to an early-stage verification run used to validate the end-to-end pipeline.



\---



\## Repository Structure



```text

intel-scene-classification/

│

├── assets/

├── configs/

├── notebooks/

├── results/

├── scripts/

├── src/

│   ├── data\_loader.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   └── gradcam.py

│

├── requirements.txt

├── setup.py

└── README.md

