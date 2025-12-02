# Aerial Image Classification on AID Dataset

## Introduction and Motivation
Image classification is a fundamental task in computer vision, playing a crucial role in perception tasks for robotics. This project focuses on the **Aerial Image Dataset (AID)**, composed of high-resolution scenes from Google Earth imagery.

As students in the robotics software field, we find this dataset particularly appropriate because aerial imagery is commonly used in autonomous systems such as UAVs for tasks like navigation, mapping, and monitoring.

This repository implements and compares two main approaches:
1.  **Classical Machine Learning:** Bag of Features (BoF) approach using local descriptors (SIFT, ORB, SURF) and various classifiers.
2.  **Deep Learning:** A custom Convolutional Neural Network (CNN) with residual connections.

## Dataset Description
The **AID dataset** contains **10,000 aerial images** with a typical resolution of $600 \times 600$ pixels, grouped into 30 classes:
*Airport, bare land, baseball field, beach, bridge, center, church, commercial, dense residential, desert, farmland, forest, industrial, meadow, medium residential, mountain, park, parking, playground, pond, port, railway station, resort, river, school, sparse residential, square, stadium, storage tanks, and viaduct.*

- **Balance:** Approximately balanced (200–400 images per category).
- **Split:** 90% Training / 10% Test.
- **Validation:** 5-fold cross-validation (80% train / 20% val).

## Methodology

### 1. Bag of Features (BoF)
We evaluated multiple local descriptors to find the optimal one for the task:
- **ORB:** Fast but showed high intra-class variability and poor performance.
- **SURF:** Robust but proprietary/commercial limitations.
- **SIFT:** Selected for the final representation due to its robustness to scale, rotation, and illumination changes.
- **AKAZE:** Good middle-ground, faster than SIFT but slightly inferior performance.

**Pipeline:**
1.  **Feature Extraction:** SIFT descriptors.
2.  **Clustering:** K-MiniBatchMeans ($k=256$) to construct the visual vocabulary.
3.  **Encoding:** Bag of Features (BoF) / VLAD.
4.  **Classification:** Evaluated multiple models.

### 2. Machine Learning Classifiers
We trained and compared the following multi-class classifiers:
- **Gaussian Naïve Bayes**
- **Softmax Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Classifier (SVC)** (Linear and RBF Kernels)

### 3. Convolutional Neural Network (CNN)
We implemented a custom CNN to contrast with classical methods.
- **Architecture:** Initial convolution layer followed by **3 residual blocks**.
- **Components:** Convolutional layers, Batch Normalization, Max-Pooling.
- **Size:** Lightweight (~1 million parameters, ~16 MB).
- **Training:** ~2 hours on an 8 GB GPU.

## Results

### Quantitative Metrics (SIFT-based BoF)

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | Overfit |
|-------|----------------|---------------|-----------|--------|----------|---------|
| **NB** | 0.5804 | 0.5080 | 0.5108 | 0.5086 | 0.4982 | 0.0724 |
| **Softmax** | 0.7344 | 0.6270 | 0.6119 | 0.6177 | 0.6060 | 0.1074 |
| **DT** | 0.6489 | 0.2755 | 0.2960 | 0.2741 | 0.2783 | 0.3734 |
| **RF** | 0.9996 | 0.5380 | 0.5347 | 0.5261 | 0.5019 | 0.4616 |
| **SVC (L)** | 0.9959 | 0.6835 | 0.6716 | 0.6781 | 0.6715 | 0.3124 |
| **SVC (RBF)**| 0.9996 | **0.6940** | 0.6883 | 0.6896 | 0.6843 | 0.3056 |

### CNN Performance
The CNN significantly outperformed all classical methods.
- **Test Accuracy:** **92.80%**
- **Macro Average:** 0.93
- **Weighted Average:** 0.93

### Discussion
- **Descriptors:** SIFT provided the best separability. ORB was too noisy.
- **ML Classifiers:** SVM (RBF) achieved the best performance among classical models (69.4%). Decision Trees and Random Forests suffered from significant overfitting.
- **Deep Learning:** The CNN achieved ~92% accuracy, demonstrating superior robustness and ability to learn features directly from data compared to the hand-crafted BoF pipeline.

## References
1. **AID Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets/data)
2. **Bag of Words:** [IBM Think Blog](https://www.ibm.com/think/topics/bag-of-words)
3. **VLAD:** Jégou et al., "Aggregating local descriptors into a compact image representation", CVPR 2010.
