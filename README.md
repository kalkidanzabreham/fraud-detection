# Fraud Detection Project

## 1. Project Overview

This project focuses on detecting fraudulent online transactions using supervised machine learning techniques.  
## 2. Dataset Description

The dataset contains transaction-level information, including:
- User demographics (e.g., age, sex)
- Transaction details (purchase value, browser, source)
- Temporal information (signup time, purchase time)
- Network-related features (IP address)
- Target variable indicating fraud (binary classification)

Fraudulent transactions are rare, leading to a **highly imbalanced dataset**.

---

## 3. Data Cleaning & Preprocessing

The following preprocessing steps were applied:

### 3.1 Handling Data Types
- Converted timestamp columns (`signup_time`, `purchase_time`) to datetime format.
- Ensured numerical columns were correctly typed for modeling.

### 3.2 Missing Values
- Checked for missing values across all features.
- No critical missing values affecting analysis were observed (or handled where necessary).

### 3.3 Sorting for Time-Based Features
- Transactions were sorted by `user_id` and `purchase_time` to enable correct time-based feature engineering.

### 3.4 Encoding Categorical Variables
- Categorical features such as `browser`, `source`, and `sex` were one-hot encoded using `pd.get_dummies`.
- `drop_first=True` was applied to avoid multicollinearity.

### 3.5 Feature Scaling
- Numerical features (`purchase_value`, `age`, `time_since_signup`) were standardized using `StandardScaler`.

---

## 4. Exploratory Data Analysis (EDA)

Key EDA findings include:

### 4.1 Class Distribution
- Fraudulent transactions represent a **small minority** of the dataset.
- This confirms the need for specialized techniques to handle class imbalance.

### 4.2 Transaction Behavior
- Fraudulent transactions tend to have:
  - Higher transaction frequency within short time windows
  - Abnormal purchasing patterns compared to legitimate users

### 4.3 Visualizations
The EDA includes:
- Class distribution plots
- Transaction value distributions
- Fraud vs non-fraud comparisons across key features

These visualizations help highlight behavioral differences between fraudulent and legitimate users.

---

## 5. Feature Engineering

Feature engineering was guided by domain logic and fraud detection best practices.

### 5.1 `time_since_signup`
- Calculated as the difference between `purchase_time` and `signup_time`.
- Rationale:
  - Fraudsters often transact shortly after account creation.
  - This feature captures suspicious early activity patterns.

### 5.2 Transaction Count in Time Windows
- A rolling 24-hour transaction count per user was computed.
- Rationale:
  - Multiple transactions in a short time window can indicate fraud.

### 5.3 IP Address Mapping
- IP addresses were mapped to higher-level groupings (e.g., regions or networks).
- Rationale:
  - Fraud often originates from unusual or inconsistent geographic/network locations.
  - Raw IPs are not directly meaningful for modeling.

### 5.4 Dropping Raw Timestamps
- Original datetime columns were removed before modeling.
- Rationale:
  - Machine learning models and SMOTE require numeric inputs.
  - Temporal information is better captured through engineered features.

---

## 6. Class Imbalance Analysis & Strategy

### 6.1 Problem Identification
- The target variable is severely imbalanced.
- Training directly on this data would bias the model toward predicting non-fraud.

### 6.2 Strategy: SMOTE
- **SMOTE (Synthetic Minority Over-sampling Technique)** was selected.
- Applied **only to the training set** to avoid data leakage.
- Used exclusively on numeric features, as SMOTE is distance-based.

### 6.3 Justification
- SMOTE creates synthetic fraud samples rather than duplicating existing ones.
- This improves model learning while preserving the original test distribution.

---

## 7. Model Building & Evaluation (Task 2)

### 7.1 Train-Test Split
- Used **Stratified Train-Test Split** to preserve class distribution.
- Separated features (`X`) and target (`y`) explicitly.

### 7.2 Baseline Model: Logistic Regression
- Implemented as an interpretable baseline.
- Used `class_weight="balanced"` to handle imbalance.
- Evaluation metrics:
  - F1-Score
  - AUC-PR
  - Confusion Matrix

### 7.3 Ensemble Model: Random Forest
- Trained a **Random Forest classifier**.
- Performed basic hyperparameter tuning:
  - `n_estimators`
  - `max_depth`
- Compared performance against baseline.

### 7.4 Cross-Validation
- Applied **Stratified K-Fold (k=5)** cross-validation.
- Reported mean and standard deviation of evaluation metrics.

### 7.5 Model Selection
- Selected the best-performing model based on:
  - AUC-PR and F1-score
  - Stability across folds
  - Interpretability vs performance trade-off

## 8. Model Explainability & Interpretation (Task 3)

### 8.1 Built-in Feature Importance
- Extracted feature importance from the Random Forest model.
- Visualized top 10 most influential features.

### 8.2 SHAP Analysis
- Generated **SHAP Summary Plot** for global feature importance.
- Generated **SHAP Force/Waterfall Plots** for:
  - True Positive (correctly detected fraud)
  - False Positive (legitimate transaction flagged as fraud)
  - False Negative (missed fraud case)

### 8.3 Key Fraud Drivers
- Time since signup
- Transaction frequency
- Purchase value
- Transaction timing (hour/day)
- IP/network-based features

### 8.4 Business Recommendations
- Additional verification for transactions shortly after signup
- Monitoring users with high transaction frequency
- Time-based fraud risk scoring
- Enhanced review for high-value transactions


## 9. Repository Structure
```bash
fraud-detection/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   └── creditcard.csv
│   └── processed/
│       └── processed_fraud_data.csv
├── models/
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
├── notebooks/
│   ├── 01_eda_fraud_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_task2.ipynb
│   └── 04_shap_explainability_task3.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline_logistic.py
│   ├── train_random_forest.py
│   ├── evaluate.py
│   └── explainability.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kalkidanzabreham/fraud-detection
   cd fraud-detection
   ```
2. Initialize the environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # For Git Bash
   pip install -r requirements.txt
    ```
3. Run notebooks or scripts
  - Start with EDA notebooks
  - Proceed to modeling and SHAP explainability

# Fraud Detection Project
**Author:** Kalkidan Abreham
