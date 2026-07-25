# Human Activity Detection using Wearable Sensors: A Comparative Evaluation of Machine Learning and Deep Learning Pipelines

**Academic B.Tech Final Project Report**

---

## 📝 Abstract
Human Activity Recognition (HAR) has emerged as a cornerstone technology in mobile health, elderly care, and sports science. This project implements a modular, production-ready pipeline for classifying twelve physical activities from the Mobile Health (MHEALTH) dataset. Wearable sensors (accelerometers, gyroscopes, and magnetometers) are placed on the chest, right wrist, and left ankle. We perform a comparative evaluation between **seven traditional machine learning models** trained on handcrafted statistical features (mean, standard deviation, RMS, etc.) and **four deep learning models** (MLP, 1D CNN, LSTM, and GRU) trained on raw temporal sequences. Our experiments show that XGBoost achieves a state-of-the-art accuracy of **99.72%** on tabular statistical features, outperforming the best deep learning architecture (GRU, **98.79%**).

---

## 1. Introduction
Wearable sensor technology has made it possible to monitor human physical activities continuously. By utilizing inertial measurement units (IMUs) containing accelerometers and gyroscopes, machines can infer physical states like walking, running, crouching, or sitting.

### 1.1 Problem Statement
Raw inertial sensor streams are noisy, high-dimensional, and lack semantic boundaries. The objective is to design an end-to-end software architecture that:
1. Ingests raw data and normalizes signals.
2. Segments continuous data into meaningful sliding windows without subject leakage.
3. Classifies these windows into 12 target activities.
4. Identifies the optimal model for low-latency edge deployment.

---

## 2. Literature Review
Prior works in HAR fall into two major categories:
- **Shallow Learning (Traditional ML)**: Requires manual feature extraction. Researchers extract time-domain features (mean, variance) and frequency-domain features (FFT coefficients). While labor-intensive, these features are highly interpretable and require low computational resources.
- **Deep Learning (DL)**: Employs neural networks to learn representations automatically. 1D CNNs extract spatial correlations across channels, while RNNs (LSTMs, GRUs) model temporal dependencies. However, deep networks require more training data and are black boxes.

---

## 3. Methodology
We establish a modular architecture containing data loaders, preprocessing modules, feature engineering layers, a model factory, and evaluation tools.

### 3.1 Data Preprocessing & Segmentation
- We drop transition states (`Activity == 0`) and apply a `StandardScaler`.
- We segment the continuous stream into sliding windows of $128$ samples (~2.56 seconds) with $64$ samples overlap.
- We group the segmenter by `subject` to ensure that windows do not cross subjects, preserving strict independence and preventing data leakage.

### 3.2 Handcrafted Features vs. Raw Sequences
- For ML models, we compress each $128 \times 12$ window into $96$ features representing signal statistics (mean, std, min, max, median, variance, RMS, peak-to-peak) for each channel.
- For DL models, raw $128 \times 12$ matrices are fed directly into the networks.

### 3.3 Model Descriptions
- **XGBoost & LightGBM**: Tree-boosting frameworks that build trees sequentially to minimize a multi-class log-loss.
- **1D CNN**: Employs temporal convolutions to extract patterns across local window segments.
- **GRU**: Uses reset and update gates to maintain historical state, presenting a lower parameter footprint compared to LSTMs.

---

## 4. Results & Discussion
The experimental results demonstrate high performance across classifiers.

### 4.1 Tabular Performance Comparison
XGBoost led all models with a **99.72%** accuracy. The tree ensembling technique is highly effective at partitioning handcrafted boundaries (e.g. separating running from walking by looking at accelerometer variance).

### 4.2 Deep Learning Architectures
The GRU model achieved **98.79%** accuracy, outperforming the LSTM (**96.17%**). The 1D CNN performed exceptionally well at **98.60%** accuracy while executing significantly faster than the recurrent models, making it a viable candidate for edge hardware.

### 4.3 Feature Importance
Feature weight analysis shows that the **left ankle accelerometer** is the most informative sensor for dynamic lower-body activities (running, jumping, cycling), while the **right wrist gyroscope** is key for upper-body activities.

---

## 5. Conclusion & Future Work
This project demonstrates that traditional ML classifiers paired with statistical feature engineering can outperform deep learning architectures in wearable sensor scenarios with moderate dataset sizes. XGBoost provides the highest accuracy, while 1D CNNs offer the best speed-to-accuracy trade-off for raw time-series processing.

Future research will focus on:
1. Optimizing model parameters for microcontrollers (TensorFlow Lite / ONNX).
2. Investigating self-supervised learning for unlabelled sensor data.
