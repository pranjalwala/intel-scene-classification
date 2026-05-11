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



<img src="assets/training\_curves.png" width="800"/>



\---



\## Confusion Matrix



<img src="assets/confusion\_matrix.png" width="700"/>



\---



\## Per-Class F1 Scores



<img src="assets/per\_class\_f1.png" width="700"/>



\---



\## Grad-CAM Visualizations



\### Buildings

<img src="assets/buildings\_gradcam.png" width="700"/>



\### Forest

<img src="assets/forest\_gradcam.png" width="700"/>



\### Glacier

<img src="assets/glacier\_gradcam.png" width="700"/>



\### Mountain

<img src="assets/mountain\_gradcam.png" width="700"/>



\### Sea

<img src="assets/sea\_gradcam.png" width="700"/>



\### Street

<img src="assets/street\_gradcam.png" width="700"/>



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



Create virtual environment:



```bash

python -m venv venv

```



Activate environment:



\### Windows



```bash

venv\\Scripts\\activate

```



\### Linux / macOS



```bash

source venv/bin/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## Usage



Download dataset:



```bash

kaggle datasets download -d puneet6060/intel-image-classification -p data

```



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

\- Git \& GitHub



\---



\## Future Improvements



\- Transfer learning with EfficientNet/ResNet50

\- TensorBoard integration

\- Hyperparameter optimization

\- Mixed precision GPU training

\- Streamlit deployment



\---



\## Author



Pranjal Wala



GitHub:  

https://github.com/pranjalwala



\---



\## License



Released under the MIT License.

