# BCI EEG Classification Experiments

This repository presents two EEG-based brain-computer interface (BCI) experiments using motor imagery data from the **BCI Competition III** datasets. The objective is to classify mental tasks using signal processing and machine learning/deep learning techniques.

---

## Included Experiments

|         Notebook File              |              Description                      |
|------------------------------------|-----------------------------------------------|
| `binary_classication.ipynb`        | Binary classification on BCI 3b dataset       |
| `multiclass_classification.ipynb`  | Multi-class classification on BCI 3a dataset  |

---

## Project Overview

### 1. Datasets
- **BCI Competition III Dataset 3b (Binary Classification)**  
  Mental tasks: Left vs Right hand motor imagery

- **BCI Competition III Dataset 3a (Multi-class Classification)**  
  Mental tasks: Left hand, Right hand, Foot, Tongue imagery

---

### 2. Preprocessing Techniques
- **Bandpass Filtering (8–30 Hz)** to isolate relevant EEG frequencies
- **Common Average Referencing (CAR)** for noise reduction
- **Epoch Extraction** aligned to task start times

---

### 3. Feature Extraction
Both statistical and spatial domain techniques:
- **Log-Variance**
- **Common Spatial Patterns (CSP)**
- **Short-Time Fourier Transform (STFT)**
- **Spectrograms (used as image inputs for CNN models)**

---

### 4. Models Used

#### Traditional Machine Learning:
- **Random Forest**
- **Support Vector Machine (SVM)**

#### Deep Learning:
- **CNN (2D/1D)** for time-frequency representations
- **CNN-RNN hybrids**
- **ChronoNet** using GRU cells
- **ResNet-50** for spectrogram-based classification

---

### 5. Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC (Binary)**
- **Confusion Matrix**
- **Cross-validation performance**

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/bci-eeg-classification.git
cd bci-eeg-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run experiments:
```bash
jupyter notebook experiment_1_bci3b_binary.ipynb
jupyter notebook experiment_2_bci3a_multiclass.ipynb
```

---

## Technologies Used

- Python 3
- NumPy, Pandas, SciPy
- MNE, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## Future Work

- Integrate hyperparameter tuning using GridSearchCV / Optuna
- Explore domain adaptation and transfer learning
- Compare EEGNet or other BCI-specific deep models
- Extend experiments to real-time BCI scenarios

---
