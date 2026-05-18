# paisabazaar-credit-score-xgboost
Credit score prediction (Good / Standard / Poor) on 100K records using XGBoost + SMOTE. Weighted F1 0.74, ROC-AUC 0.87, 3-fold CV F1 0.77. Top risk drivers via SHAP: Credit Mix, Interest Rate, Outstanding Debt. Python · XGBoost · MLflow
# 🏦 Paisabazaar — Banking Credit Score Prediction

> Predicts customer credit score (Good / Standard / Poor) using XGBoost + SMOTE on 100K records. Weighted F1 0.74, ROC-AUC 0.87. Risk drivers explained via SHAP.

---

## 📌 Problem Statement
Credit scoring determines whether a customer qualifies for a loan and at what interest rate. Manual scoring is slow and inconsistent. This project builds an automated ML pipeline to classify customers into Good, Standard, or Poor credit score categories — and explains *why* using SHAP.

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset size | 100,000 rows × 28 columns |
| Model | XGBoost + SMOTE |
| Weighted F1 | **0.74** |
| Weighted ROC-AUC | **0.87** |
| 3-Fold CV F1 | **0.77 ± 0.03** |
| Top risk drivers | Credit Mix, Interest Rate, Outstanding Debt |

## 🛠️ Tech Stack
`Python` · `XGBoost` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `SHAP` · `MLflow` · `Pandas` · `Matplotlib` · `Seaborn`

## 📁 Project Structure
```
Paisabazaar_Credit_Score/
│
├── Paisabazaar_Banking_Fraud_Complete.ipynb   ← Main notebook
├── dataset-2.csv                               ← Dataset (place here)
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
# 1. Install dependencies (run once)
pip install -r requirements.txt

# 2. Place dataset-2.csv in this folder
# 3. Open notebook
jupyter notebook Paisabazaar_Banking_Fraud_Complete.ipynb

# 4. Kernel → Restart & Run All
# Expected runtime: 5–8 minutes
```

## 📂 Dataset
- **Source:** [Kaggle — Credit Score Classification Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **Target:** Credit_Score (Good / Standard / Poor)
- **Challenge:** Class imbalance (Standard 53%, Poor 29%, Good 18%)

## 🔍 Pipeline Steps
1. Exploratory Data Analysis (EDA) — 30+ visualizations
2. Drop PII columns (ID, SSN, Name)
3. Label encode categorical features
4. SMOTE to balance classes (80K → 127K training rows)
5. XGBoost training and evaluation
6. Confusion matrix + ROC-AUC
7. 3-fold cross-validation
8. SHAP explainability (top risk drivers)
9. MLflow logging
10. Streamlit app generation

## 🔍 Key Findings
- Credit Mix (type of credit accounts) is the strongest predictor
- Customers with high Outstanding Debt + high Interest Rate → consistently Poor score
- SMOTE significantly improved recall on the minority "Good" class
- CV F1 of 0.77 confirms the model generalises — not just memorising training data

---
*Project completed as part of AI/ML Internship at Labmentix Pvt. Ltd. (2025–2026)*
