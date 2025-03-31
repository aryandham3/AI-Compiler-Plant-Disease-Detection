# AI-Compiler-Plant-Disease-Detection
AI-compiler for image-based disease detection in plants using Hybrid CNN + GWO. The model classifies tomato leaf infections with deep learning and optimization techniques.


Overview
This project focuses on detecting leaf infections in plants using a Hybrid CNN + Grey Wolf Optimizer (GWO) model. The dataset consists of labeled images of plant leaves with various diseases. The deep learning model is trained to classify these images accurately, helping in early disease detection.

Dataset
The dataset consists of Tomato plant leaf images, categorized into different disease types. It contains 6065 training images and 1514 validation images. The dataset is stored as a ZIP file (CD.zip), which needs to be extracted before training.

Implementation
Data Preprocessing
Images are loaded and normalized.
Data is split into training (80%) and validation (20%) sets.

Model Training
A CNN model is designed with optimized convolutional layers.
The model is trained on the dataset for 10 epochs.
The trained model is saved as tomato_disease_model.h5.

Training Output:
Epoch 1: Accuracy - 45.32%, Loss - 1.3172
Epoch 5: Accuracy - 90.38%, Loss - 0.2636
Epoch 10: Accuracy - 94.96%, Loss - 0.1343
Final Validation Accuracy: 94.65%

Performance Evaluation
The model's accuracy and loss are tracked over epochs.
Validation accuracy is used to check generalization.
Metrics such as precision, recall, and F1-score will be calculated in future work.

Future Work
Optimize CNN architecture:
Improve depth of convolutional layers and filter sizes for better feature extraction.
Use pre-trained architectures like ResNet or EfficientNet combined with GWO for better results.

Enhance Dataset:
Train on larger datasets with diverse plant diseases to improve model robustness.

Performance Metrics:
Measure accuracy, precision, recall, and F1-score using the following formulas:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-score = 2 × (Precision × Recall) / (Precision + Recall)
Present results in both tabular and graphical formats.
Implement an innovative fitness function to improve classification accuracy.

Integration of GWO:
Implement Grey Wolf Optimizer (GWO) for optimizing hyperparameters.
Improve model efficiency and convergence rate using GWO-based tuning.

How to Use
Extract the dataset (CD.zip).
Open Jupyter Notebook and run the training script.
The trained model will be saved as tomato_disease_model.h5.

Requirements
Python
Jupyter Notebook
TensorFlow/Keras
NumPy
Matplotlib
OpenCV


Contributors

Aryan Dham (147)
Aayush Kakkar (089)
