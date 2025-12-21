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

## 7. Repository Structure
```bash
fraud-detection/
├── .github/
│   └── workflows/          # CI/CD pipelines (e.g., unittests.yml)
├── .vscode/                # Editor configurations
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for analysis
│   ├── eda-creditcard.ipynb
│   ├── eda-fraud-data.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── scripts/                # Modular Python scripts
├── src/                    # Source code and helper functions
├── tests/                  # Unit tests for the project
├── .gitignore              # Files to be ignored by git
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
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

