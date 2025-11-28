
---

## 🧠 Problem Statement

Financial institutions need to determine **whether a borrower is likely to default**.  
Traditional ML models often act as “black boxes,” making them unsuitable for regulated environments.

ExCre solves this problem by combining:

- **High-performance ML models**
- **Transparent, interpretable explanations**
- **Cost-sensitive evaluation metrics (recall-focused)**

---

## 🛠️ Tech Stack

- **Programming:** Python  
- **ML Models:** Logistic Regression, Random Forest, XGBoost  
- **Explainability:** SHAP  
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook  

---

## 🔍 Workflow Overview

### 1️⃣ Exploratory Data Analysis (EDA)
- Visualized distributions, correlations, and risk segments  
- Identified missing values & outliers  
- Analyzed class imbalance in target variable

### 2️⃣ Data Preprocessing
- Treated missing values  
- One-hot encoded categorical variables  
- Standardized numerical features  
- Train-test split for robust evaluation  

### 3️⃣ Model Training & Comparison
Models evaluated:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- **XGBoost** (best performer)

### 4️⃣ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  
- ROC & AUC Score  

**Best Model (XGBoost):**  
- **Accuracy:** 0.88  
- **Precision:** 0.85  
- **Recall:** 0.91  
- **AUC-ROC:** 0.94  

### 5️⃣ Explainability (SHAP)
- Global feature importance (beeswarm plot)  
- Local explanations (force plot)  
- Key predictors identified:
  - Loan Amount  
  - Applicant Income  
  - Credit History  
  - Debt-to-Income Ratio  

Through SHAP, the model becomes **transparent**, allowing auditors and risk officers to understand *why* a prediction is made.

---

## 📈 Visual Outputs (Optional)
Add images such as:
- ROC Curve  
- Confusion Matrix  
- SHAP Summary Plot  
- SHAP Force Plot  

Place them inside `/plots` and embed in README.

---

## 🧪 Results Summary

| Model                  | Accuracy | Precision | Recall | AUC-ROC |
|------------------------|----------|-----------|--------|---------|
| Logistic Regression    | 0.78     | 0.74      | 0.81   | –       |
| Random Forest          | 0.86     | 0.83      | 0.89   | –       |
| **XGBoost (Best)**     | **0.88** | **0.85**  | **0.91**| **0.94** |

---

## 🔮 Future Improvements

- Hyperparameter tuning with Optuna or Hyperopt  
- Deploying model as REST API (FastAPI)  
- Streamlit dashboard for real-time credit scoring  
- Incorporating additional financial features  
- Handling data imbalance with SMOTE or class weighting  

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributions

Pull requests, improvements, or suggestions are welcome!

---

## 📧 Contact  

**Author:** Abdul Rahim  
GitHub: https://github.com/abdulrahim200  
Project Link: https://github.com/abdulrahim200/ExCre--An-Explainable-Credit-Risk-Prediction-Model-
