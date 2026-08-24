# Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom company using the Telco Customer Churn dataset. The notebook covers data cleaning, exploratory data analysis, feature engineering, and training/evaluating four classification models to identify customers likely to leave.

## Dataset

**Telco Customer Churn** — 7,043 customers, 21 original columns (19 features + `customerID` + `Churn` target).

Features include customer demographics (senior citizen, partner, dependents), account info (tenure, contract type, payment method, billing), and subscribed services (phone, internet, streaming, tech support, etc.).

`customerID` and `gender` were dropped early on, as they carry no predictive relationship to churn.

## Project Structure (Notebook Parts)

### Part 1 — Data Cleaning & Exploratory Data Analysis
- Dropped irrelevant columns (`customerID`, `gender`)
- Handled missing/blank values in `TotalCharges` (11 blank entries filled with the column mode)
- Visualized key relationships:
  - Churn by contract type
  - Churn by internet service
  - Monthly charges distribution
  - Tenure vs. churn
- Converted `TotalCharges` to numeric and applied a square-root transform to reduce skew (0.963 → 0.309)
- Encoded categorical variables:
  - Binary mapping (Yes/No → 1/0) for `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`
  - One-hot encoding (`pd.get_dummies`, drop-first) for `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`
- Reviewed correlation of all features against `Churn` via a heatmap
- Split data into train/test sets (80/20, `random_state=42`) and scaled features with `StandardScaler`

### Part 2 — Model Training & Evaluation
Four classification models were trained and compared:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8126 | 0.6799 | 0.5523 | 0.6095 | 0.8607 |
| KNN (best k=9) | 0.7928 | 0.6216 | 0.5550 | 0.5864 | 0.8156 |
| Decision Tree (best params) | 0.7991 | 0.6510 | 0.5201 | 0.5782 | 0.8277 |
| Random Forest | 0.8112 | 0.6890 | 0.5228 | 0.5945 | 0.8590 |

- **Logistic Regression**: baseline linear classifier using a sigmoid function to produce churn probabilities
- **KNN**: tuned over k = 3, 5, 7, 9 (odd values); best result at k=9
- **Decision Tree**: grid search over `min_samples_leaf` (5, 10, 20) and `max_depth` (5, 7, 10); best combination was `min_samples_leaf=10, max_depth=7`, giving a healthy train/test accuracy gap (no overfitting)
- **Random Forest**: 100 trees, `max_depth=10`, `max_features="sqrt"`; the top-performing model overall

### Part 3 — Model Comparison
- Compared all four models on confusion matrices (TN/FP/FN/TP), accuracy, precision, recall, F1 score, and ROC-AUC
- **Random Forest** delivers the best overall balance of metrics, followed closely by Logistic Regression

### Part 4 — Discussion & Conclusion
- **Best model**: Random Forest, for its balance of accuracy, precision, recall, F1, and ROC-AUC, and its ability to capture complex patterns via multiple trees
- **Precision vs. recall**: Recall is prioritized for this business problem — missing a customer who is likely to churn (a false negative) means a lost retention opportunity, which is costlier than a false positive
- **Interpretability vs. performance**: Decision Trees are easier to explain as simple rule sequences, while Random Forest trades some interpretability for stronger, more stable predictive performance

## Key Findings
- Contract type, tenure, monthly charges, and service-related features are the strongest churn indicators
- Month-to-month contracts and fiber optic internet customers show higher churn rates
- Random Forest and Logistic Regression are the top performers by ROC-AUC (0.86 and 0.861, respectively)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

1. Place `Telco-Customer-Churn.csv` in the expected data path (update the path in the notebook if not running on Kaggle)
2. Run the notebook cells in order (Part 1 → Part 4)
3. Review the model comparison table and confusion matrices to evaluate performance

## Notes
- Original data path used: `/kaggle/input/datasets/abbas829/telco-customer-churn-dataset/Telco-Customer-Churn.csv` — update this if running locally or outside Kaggle
