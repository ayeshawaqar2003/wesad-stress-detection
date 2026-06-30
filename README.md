<div align="center">

# 🧠 WESAD Stress Detection System

### Subject-Independent Physiological Stress Classification using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A **subject-independent stress detection system** trained on the **WESAD** dataset using chest-worn physiological signals.

The model classifies human physiological states into:

🟢 **Baseline** • 🔴 **Stress** • 🟡 **Amusement** • 🔵 **Meditation**

## **🏆 Best Accuracy: 71.52%**

Achieved using an **ExtraTrees Ensemble (1500 Trees)** with **Quantile-Normal Transformation** and **SMOTE-Tomek** balancing.

</div>

---

# 📖 Overview

Physiological stress detection is challenging because every person's biological signals are different.

A heart rate or electrodermal activity (EDA) level indicating stress in one individual may represent a normal resting state for another. Traditional machine learning models often perform well only on people seen during training and fail to generalize to new subjects.

This project addresses that challenge by applying **Quantile-Normal Transformation** before feature extraction, normalizing each subject's physiological distribution into a common statistical space. This significantly reduces inter-subject variability and enables **subject-independent stress classification**.

---

# ✨ Features

- 📂 Loads raw WESAD `.pkl` files
- 📈 Quantile-Normal Transformation
- 🧮 30 statistical features extracted
- ⏱ 10-second sliding windows with 50% overlap
- 🌲 ExtraTrees Ensemble classifier
- ⚖️ SMOTE-Tomek class balancing
- 👤 Subject-independent evaluation
- 📊 Interactive Streamlit dashboards

---

# 📂 Dataset

### WESAD (Wearable Stress and Affect Detection)

A publicly available multimodal dataset for stress recognition.

| Property | Value |
|-----------|-------|
| Subjects | 15 (S2–S17 excluding S12) |
| Sensor | Chest-worn RespiBAN |
| Sampling Rate | 700 Hz |
| Signals Used | EDA, Temperature, Respiration |
| Classes | Baseline, Stress, Amusement, Meditation |

> **Dataset is NOT included** due to licensing restrictions.

Download it from:

https://uni-siegen.de/life/forschung/telekom_stiftung/ubisys/tools/wesad.html

---

# ⚙️ Methodology

## Signal Processing Pipeline

```text
Raw WESAD Signals
        │
        ▼
Quantile-Normal Transformation
        │
        ▼
Sliding Window (10 sec, 50% overlap)
        │
        ▼
Feature Extraction
(30 Statistical Features)
        │
        ▼
SMOTE-Tomek Balancing
        │
        ▼
ExtraTrees Classifier
        │
        ▼
Stress Prediction
```

---

# 🧮 Feature Extraction

For each signal (EDA, Temperature, Respiration), the following statistics are extracted:

- Mean
- Standard Deviation
- Variance
- Minimum
- Maximum
- Median
- 25th Percentile
- 75th Percentile
- Skewness
- Kurtosis

**Total Features = 30**

---

# 🤖 Model Comparison

| Model | Trees | Max Depth | Features | Class Weight | Accuracy |
|--------|-------|-----------|-----------|--------------|----------|
| Model 1 | 1200 | 40 | sqrt | balanced | 70.98% |
| Model 2 | 2000 | 50 | All (30) | balanced_subsample | 70.10% |
| ⭐ Model 3 | 1500 | 45 | log2 | balanced | **71.52%** |

**Final selected model:** **Model 3**

---

# 📈 Classification Report

```text
              precision    recall    f1-score

Baseline         0.87       0.83       0.85
Stress           0.68       0.81       0.74
Amusement        0.33       0.12       0.18
Meditation       0.61       0.74       0.67

Overall Accuracy: 71.52%
```

### Observations

✅ Excellent Baseline detection

✅ Strong Stress detection

⚠️ Amusement remains difficult because its physiological signatures overlap with Baseline.

Future work includes frequency-domain features such as **Power Spectral Density (PSD)**.

---

# 📊 Accuracy Progression

| Stage | Method | Accuracy |
|-------|--------|----------|
| Baseline | Random Forest + Z-score | ~61% |
| Without Subject Normalization | Multi-signal | 38.91% |
| Iteration A | ExtraTrees + Quantile | 70.98% |
| Iteration B | ExtraTrees + SMOTE-Tomek | 70.10% |
| 🏆 Final | ExtraTrees + Quantile + SMOTE-Tomek | **71.52%** |

---

# 🖥️ Applications

This repository includes two Streamlit applications.

## 🌿 app.py

### Personal Wellness Monitor

Designed for general users.

Features:

- Plain-language physiological inputs
- Confidence indicator
- Easy-to-understand predictions

---

## 📊 app2.py

### Research Dashboard

Designed for researchers.

Includes:

- Numerical physiological inputs
- Feature importance visualization
- Model details
- Methodology overview

---

# 📁 Project Structure

```text
wesad-stress-detection/
│
├── train_model.py
├── train_model.3.py
├── train_model.4.py
│
├── app.py
├── app2.py
│
├── graphs.py
├── requirements.txt
├── README.md
│
├── wesad_stress_model.pkl
│
└── WESAD/
    ├── S2/
    ├── S3/
    └── ...
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/ayeshawaqar2003/wesad-stress-detection.git

cd wesad-stress-detection
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Download WESAD and place it in the project root:

```text
WESAD/
    S2/
    S3/
    ...
```

---

## Train the Model

```bash
python train_model.py
```

This generates:

```text
wesad_stress_model.pkl
```

---

## Run the Streamlit Dashboard

### User Dashboard

```bash
streamlit run app.py
```

### Research Dashboard

```bash
streamlit run app2.py
```

---

# 💡 Key Technical Decisions

<details>

<summary><strong>Why Quantile-Normal Transformation?</strong></summary>

Different individuals exhibit vastly different physiological baselines.

Quantile transformation maps each subject's signals to a standard normal distribution, dramatically reducing inter-subject variability.

This proved to be the single largest contributor to improved subject-independent accuracy.

</details>

<details>

<summary><strong>Why ExtraTrees?</strong></summary>

ExtraTrees introduces additional randomness compared to Random Forest by selecting split thresholds randomly.

Benefits include:

- Lower variance
- Reduced overfitting
- Better performance on medium-sized physiological datasets

</details>

<details>

<summary><strong>Why SMOTE-Tomek?</strong></summary>

The WESAD dataset is highly imbalanced.

SMOTE-Tomek simultaneously:

- Oversamples minority classes
- Removes noisy boundary samples

leading to improved classifier robustness.

</details>

<details>

<summary><strong>Why Subject-Independent Validation?</strong></summary>

A clinically useful stress detector must generalize to unseen individuals.

Subjects **S6–S9** were completely excluded from training and used solely for testing.

</details>

---

# 📄 Research Paper

This work is documented in the accompanying research paper:

> **Non-Diagnostic Human Health State Recognition Using Physiological Wearable Data and Machine Learning**

The paper includes:

- Literature Review
- Signal Processing
- Feature Engineering
- Quantile Normalization
- Model Development
- Experimental Results
- Error Analysis

📄 Included in this repository as **Research Paper.pdf**

**Co-authored with Ayesha Qazi**

---

# 🔮 Future Work

- ⌚ Incorporate wrist-worn sensors (BVP, ACC)
- 🤖 Transformer-based architectures
- 📡 Real-time wearable inference
- ❤️ Continuous affect monitoring
- 📈 Frequency-domain feature extraction

---


---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>



 
## Author
 
**Ayesha Waqar** — Electronics Engineering graduate, Master's student specializing in AI

**Ayesha Qazi** — BIO Engineering graduate, Master's student specializing in AI

 
📧 ayeshagul2003@hotmail.com · 💼 www.linkedin.com/in/ayeshagulwaqar · 🐙 [github.com/ayeshawaqar2003](https://github.com/ayeshawaqar2003)
