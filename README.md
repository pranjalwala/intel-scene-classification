\# Intel Scene Classification using Residual CNNs



Research-oriented scene classification pipeline using a custom ResNet-style CNN built with TensorFlow/Keras for multi-class natural scene understanding.



\---



\## Overview



This project implements a modular deep learning workflow for classifying natural scenes from the Intel Image Classification dataset (\~25,000 images across 6 classes).



\### Features



\- Custom ResNet-style CNN with residual connections

\- TensorFlow `tf.data` pipeline

\- Data augmentation and preprocessing

\- Grad-CAM explainability

\- Confusion matrix and F1-score evaluation

\- Config-driven experimentation

\- Modular training and evaluation scripts



\---



\## Dataset



\*\*Intel Image Classification Dataset\*\*



Classes:

\- Buildings

\- Forest

\- Glacier

\- Mountain

\- Sea

\- Street



Dataset Source:  

https://www.kaggle.com/datasets/puneet6060/intel-image-classification



\---



\## Model Architecture



The model includes:



\- Residual skip connections

\- 1×1 projection shortcuts

\- Batch normalization

\- Dropout regularization

\- Global average pooling

\- Softmax classification head



\---



\# Results



\## Training Curves



<p align="center">

&#x20; <img src="assets/training\_curves.png" width="800">

</p>



\---



\## Confusion Matrix



<p align="center">

&#x20; <img src="assets/confusion\_matrix.png" width="700">

</p>



\---



\## Per-Class F1 Scores



<p align="center">

&#x20; <img src="assets/per\_class\_f1.png" width="700">

</p>



\---



\## Grad-CAM Visualizations



\### Buildings



<p align="center">

&#x20; <img src="assets/buildings\_gradcam.png" width="700">

</p>



\### Forest



<p align="center">

&#x20; <img src="assets/forest\_gradcam.png" width="700">

</p>



\### Glacier



<p align="center">

&#x20; <img src="assets/glacier\_gradcam.png" width="700">

</p>



\### Mountain



<p align="center">

&#x20; <img src="assets/mountain\_gradcam.png" width="700">

</p>



\### Sea



<p align="center">

&#x20; <img src="assets/sea\_gradcam.png" width="700">

</p>



\### Street



<p align="center">

&#x20; <img src="assets/street\_gradcam.png" width="700">

</p>



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

```



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/pranjalwala/intel-scene-classification.git

cd intel-scene-classification

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## Usage



Train model:



```bash

python src/train.py

```



Evaluate model:



```bash

python src/evaluate.py

```



Generate Grad-CAM visualizations:



```bash

python src/gradcam.py

```



\---



\## Technologies Used



\- TensorFlow / Keras

\- NumPy

\- Pandas

\- scikit-learn

\- Matplotlib

\- Jupyter Notebook



\---



\## Future Improvements



\- Transfer learning with EfficientNet

\- Hyperparameter optimization

\- TensorBoard integration

\- Mixed precision training

\- Streamlit deployment



\---



\## Author



Pranjal Wala



GitHub:  

https://github.com/pranjalwala

