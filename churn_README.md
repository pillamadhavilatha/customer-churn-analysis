# 📉 Customer Churn Prediction

> **Data Analyst Internship Project** — Predicting telecom customer churn using exploratory data analysis and a Random Forest classifier on the IBM Telco dataset.

---

## 📌 Project Overview

Customer churn — when customers stop using a service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds an end-to-end machine learning pipeline to **predict which customers are likely to churn**, enabling the business to take proactive retention action.

The project covers the full data science lifecycle: data cleaning, exploratory analysis, feature encoding, model training, evaluation, and feature importance analysis.

---

## 🗂️ Repository Structure

```
customer-churn-prediction/
│
├── customerChurn.ipynb                       # Main Jupyter Notebook
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
└── images/                                   # Saved plot outputs
    ├── churn_distribution.png
    ├── contract_vs_churn.png
    ├── tenure_distribution.png
    ├── monthly_charges_boxplot.png
    ├── correlation_heatmap.png
    └── feature_importance.png
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` — `Yes` = churned, `No` = retained |
| **Churn Rate** | ~26.5% |

> ⚠️ The dataset is not included in this repository. Download it from the Kaggle link above and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

### Feature Glossary

| Feature | Description |
|---------|-------------|
| `tenure` | Number of months with the company |
| `MonthlyCharges` | Current monthly bill amount |
| `TotalCharges` | Total amount charged to date |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer / Credit card |
| `InternetService` | DSL / Fiber optic / No |
| `TechSupport` | Whether customer has tech support add-on |
| `OnlineSecurity` | Whether customer has online security add-on |
| `Churn` | **Target** — did the customer leave? |

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Visual Insights

**1. Churn Distribution**
- ~26.5% of customers churned — a moderate class imbalance

**2. Contract Type vs Churn**
- Customers on **month-to-month** contracts churn at a dramatically higher rate
- Customers on **two-year contracts** rarely churn — long-term commitment reduces risk significantly

**3. Payment Method vs Churn**
- **Electronic check** users show the highest churn rate
- Automatic payment methods (bank transfer, credit card) correlate with lower churn

**4. Tenure Distribution**
- New customers (low tenure) are far more likely to churn
- Churn drops significantly after ~12 months; long-tenured customers (50+ months) are very loyal

**5. Monthly Charges vs Churn**
- Churned customers have **noticeably higher monthly charges**
- Price sensitivity is a key churn driver

---

## ⚙️ Methodology

### 1. Data Cleaning
- Converted `TotalCharges` from object to numeric (`errors='coerce'` — 11 blank values became NaN)
- Filled missing `TotalCharges` with the **column median**

### 2. Feature Encoding
- Applied `LabelEncoder` to all categorical columns for model compatibility

### 3. Feature / Target Split
```
X  →  All columns except 'Churn'  (20 features)
y  →  'Churn' label-encoded as 0 (No) / 1 (Yes)
```

### 4. Train-Test Split
```
Train: 80%  |  Test: 20%  |  stratify=y  (preserves class ratio)
```

### 5. Model — Random Forest Classifier
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~80–82% |

> 💡 Note: With class imbalance (~26.5% churn), Precision, Recall, F1 Score, and ROC-AUC are more meaningful than accuracy alone. See the Future Improvements section below.

### Top 10 Features Driving Churn

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `TotalCharges` | Lifetime customer value |
| 2 | `tenure` | How long the customer has stayed |
| 3 | `MonthlyCharges` | Current billing amount |
| 4 | `Contract` | Commitment level |
| 5 | `PaymentMethod` | Payment behavior signal |
| 6 | `OnlineSecurity` | Value-added service usage |
| 7 | `TechSupport` | Service engagement indicator |
| 8 | `InternetService` | Service type |
| 9 | `PaperlessBilling` | Digital engagement |
| 10 | `SeniorCitizen` | Demographic segment |

---

## 💡 Key Business Insights

1. **Target month-to-month customers first** — highest churn risk; a contract upgrade incentive could have major impact.

2. **Price sensitivity is real** — customers with high monthly charges churn more; targeted discounts or bundle offers may help.

3. **New customers need attention** — the first 12 months are critical. An onboarding loyalty program could reduce early churn significantly.

4. **Electronic check users are a red flag** — this payment method strongly correlates with churn. Incentivizing auto-pay enrollment could retain these customers.

5. **Value-added services matter** — customers without `OnlineSecurity` or `TechSupport` churn more; upselling these services can double as a retention strategy.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | EDA visualizations |
| Scikit-learn | Encoding, modeling, evaluation |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), place it in the project root, then update the path in the notebook:
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### 4. Launch the notebook
```bash
jupyter notebook customerChurn.ipynb
```

---

## 🔮 Future Improvements

- [ ] Apply **SMOTE** to handle class imbalance and improve recall on churners
- [ ] Add **Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC** evaluation
- [ ] Test additional models: **Logistic Regression**, **XGBoost**, **LightGBM**
- [ ] **Hyperparameter tuning** with GridSearchCV or RandomizedSearchCV
- [ ] Build an **interactive churn risk dashboard** using Streamlit
- [ ] Deploy model as a **REST API** with Flask or FastAPI

---

## 📝 Conclusion

This project demonstrates how machine learning can solve a critical real-world business problem — predicting customer churn in the telecom industry. Through EDA, we identified the key churn drivers (contract type, tenure, monthly charges, payment method), and a **Random Forest classifier** achieved ~80–82% accuracy on unseen data.

The feature importance analysis provides directly actionable insights for a customer retention team, making this a practical, business-aligned data science project.

---

## 🙋 Author

**[Your Name]**  
Data Analyst Intern  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
