# Experimental Results & Performance Analysis

This document reports the performance results obtained by training and evaluating the 11 classifiers on the MHEALTH dataset split.

---

## 🏆 Model Performance Summary

The classifiers are ranked below by their test accuracy on the 20% stratified holdout set (1,070 test windows):

| Rank | Model | Accuracy | F1 (Macro) | F1 (Weighted) | Family |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | **0.9972** | **0.9965** | **0.9972** | Machine Learning |
| 🥈 | **LightGBM** | **0.9953** | **0.9939** | **0.9953** | Machine Learning |
| 🥉 | **Random Forest** | **0.9953** | **0.9956** | **0.9953** | Machine Learning |
| 4 | **CatBoost** | 0.9953 | 0.9956 | 0.9953 | Machine Learning |
| 5 | **Logistic Regression** | 0.9935 | 0.9921 | 0.9935 | Machine Learning |
| 6 | **Support Vector Machine (SVM)** | 0.9897 | 0.9903 | 0.9897 | Machine Learning |
| 7 | **GRU (Gated Recurrent Unit)** | 0.9879 | 0.9885 | 0.9878 | Deep Learning |
| 8 | **Decision Tree** | 0.9879 | 0.9860 | 0.9879 | Machine Learning |
| 9 | **1D CNN (Convolutional)** | 0.9860 | 0.9868 | 0.9860 | Deep Learning |
| 10 | **MLP (Multi-Layer Perceptron)** | 0.9701 | 0.9681 | 0.9699 | Deep Learning |
| 11 | **LSTM (Long Short-Term Memory)** | 0.9617 | 0.9592 | 0.9610 | Deep Learning |

---

## 🔍 Key Insights

### 1. Traditional ML vs. Deep Learning
- **Traditional ML** classifiers trained on 96 statistical features outperform deep learning architectures trained on raw sequences. XGBoost achieved the highest test accuracy of **99.72%**, misclassifying only 3 out of 1,070 test windows.
- Hand-crafted statistical features (specifically variance, standard deviation, and peak-to-peak amplitude) provide clean, direct markers of movement intensity that trees can partition easily.

### 2. Deep Learning Behavior
- The **GRU** model achieved the highest accuracy among deep learning models (**98.79%**), followed closely by the **1D CNN** (**98.60%**).
- GRU outperformed LSTM (**96.17%**). GRU's simpler architecture (having only update and reset gates) proved easier to optimize on this sequence length (128 samples) with less risk of overfitting.
- The 1D CNN was highly efficient, utilizing spatial-temporal kernels to capture features without the recurrence bottleneck.

### 3. Feature Importance Analysis
Analysis of the tree-based models (XGBoost, LightGBM, Random Forest) reveals:
- **Ankle Sensors** (particularly accelerometer variance and peak-to-peak values on the Y and Z axes) were highly predictive of dynamic classes like *Cycling*, *Jogging*, and *Running*.
- **Wrist Sensors** (gyroscope variance and peak-to-peak values) were essential for detecting *Waist bends forward* and *Frontal elevation of arms*.
- **Static vs. Dynamic signals**: Low variance in all sensors was the primary indicator for static classes (*Standing still*, *Sitting*, *Lying down*).

---

## 📈 Figures and Artifacts

The execution pipeline automatically saves the following charts to `outputs/figures/`:
- `model_comparison.png`: Accuracy bar chart comparing all 11 models.
- `confusion_matrix_<model>.png`: Normalized confusion heatmaps showing misclassification clusters.
- `roc_<model>.png`: ROC curves representing One-vs-Rest AUC performance.
- `training_curves_<model>.png`: Loss and accuracy curves over training epochs for MLP, CNN1D, LSTM, and GRU.
