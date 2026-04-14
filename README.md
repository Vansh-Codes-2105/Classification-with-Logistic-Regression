# 🧠 Breast Cancer Classification using Logistic Regression

## 📖 Overview

This project implements a **Logistic Regression model** to perform **binary classification** on the Breast Cancer Wisconsin dataset. The objective is to predict whether a tumor is **malignant** or **benign** based on diagnostic features.

The workflow follows a standard machine learning pipeline including preprocessing, training, evaluation, and threshold optimization.

---

## 🎯 Objectives

* Build a reliable binary classification model
* Evaluate model performance using standard metrics
* Understand the impact of decision thresholds
* Interpret key concepts such as sigmoid function and ROC-AUC

---

## 📂 Dataset Information

* **Dataset**: Breast Cancer Wisconsin Dataset

* **Target Variable**: `diagnosis`

  * Malignant (M) → 1
  * Benign (B) → 0

* **Features**:

  * Radius, texture, perimeter, area, smoothness, etc.

---

## 🏗️ Project Structure

```bash id="o9f3ps"
├── data.csv
├── main.py
├── README.md
```

---

## ⚙️ Tech Stack

* **Language**: Python
* **Libraries**:

  * pandas
  * numpy
  * scikit-learn

---

## 🔄 Methodology

### 1. Data Preprocessing

* Removed irrelevant columns (e.g., `id`)
* Encoded categorical labels into numerical format

### 2. Train-Test Split

* Split dataset into **80% training** and **20% testing**

### 3. Feature Scaling

* Applied **StandardScaler** to normalize feature values

### 4. Model Training

* Used **Logistic Regression** for classification

### 5. Model Evaluation

* Confusion Matrix
* Precision
* Recall
* ROC-AUC Score

### 6. Threshold Optimization

* Adjusted classification threshold to improve recall/precision trade-off

---

## 📊 Performance Metrics

| Metric           | Description                      |
| ---------------- | -------------------------------- |
| Confusion Matrix | Classification summary           |
| Precision        | Accuracy of positive predictions |
| Recall           | Ability to detect positives      |
| ROC-AUC          | Overall model performance        |

---

## 📈 Key Concept

### Sigmoid Function

The logistic regression model uses the sigmoid function to map predictions to probabilities:

σ(z) = 1 / (1 + e⁻ᶻ)

* Output range: (0, 1)
* Enables binary classification

---

## ▶️ Installation & Execution

### 1. Clone the Repository

```bash id="5j81zk"
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash id="m7o3qk"
pip install pandas numpy scikit-learn
```

### 3. Run the Project

```bash id="q0txfk"
python main.py
```

---

## 📌 Sample Results

* High **ROC-AUC score (~0.99)** indicating strong model performance
* Balanced precision and recall after threshold tuning

---

## 🚀 Applications

* Medical diagnosis systems
* Fraud detection
* Spam classification

---

## ⚠️ Limitations

* Sensitive to class imbalance
* Assumes linear relationship between features and log-odds

---

## 🔮 Future Improvements

* Implement cross-validation
* Use advanced models (Random Forest, XGBoost)
* Add visualization (ROC Curve, Precision-Recall Curve)

---

## 👤 Author

* **Krishi**
* Add your GitHub profile link here

---

## 📜 License

This project is open-source and available under the MIT License.
