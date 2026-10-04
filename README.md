# fraud-detection-pipeline
# Fraud Detection Pipeline

Supervised ML pipeline for credit card fraud detection on a highly imbalanced dataset (93,948 transactions, 0.23% fraud).

- **Methods:** StandardScaler + SMOTE oversampling inside an imblearn Pipeline; Logistic Regression and Random Forest compared; Random Forest tuned with GridSearchCV (3-fold stratified CV)
- **Results (test set):**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.097 | 0.861 | 0.175 | 0.981 |
| Random Forest | 0.872 | 0.791 | 0.829 | 0.973 |
| Random Forest (tuned) | 0.739 | 0.791 | 0.764 | 0.975 |

- **Takeaway:** Logistic Regression catches the most fraud but flags many false positives. Random Forest gives the best precision/recall balance.
- **Tech:** Python, scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn
