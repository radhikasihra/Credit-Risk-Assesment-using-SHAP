# Explainable Credit Risk Assessment

**Machine Learning · XGBoost · SHAP · Probability Calibration · FastAPI**

This project predicts the risk associated with a loan application from applicant and loan characteristics. It compares an interpretable Logistic Regression baseline with XGBoost, examines prediction errors, and explores explanations for individual decisions. A FastAPI endpoint is included for model inference.

## Project snapshot

| Area | Details |
| --- | --- |
| Data | 32,581 loan records in the supplied CSV; `loan_status` is the binary target |
| Inputs | Age, income, home ownership, employment length, loan purpose and grade, amount, interest rate, loan-to-income ratio, prior default flag, and credit history length |
| Preparation | Duplicate and outlier checks; numeric imputation; categorical imputation and one-hot encoding; scaling for Logistic Regression |
| Modeling | Class-weighted Logistic Regression baseline and XGBoost with imbalance weighting |
| Validation | Stratified 80/20 split, five-fold cross-validation, and randomized hyperparameter search |
| Interpretation | SHAP summary and applicant-level waterfall plots; false-positive and false-negative review |
| Integration | Saved model artifacts and a FastAPI `POST /predict` endpoint returning a probability and risk category |

## Recorded performance

The notebook reports the following results on its held-out test split at the models' default prediction thresholds:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.82 | 0.56 | 0.79 | 0.65 |
| Tuned XGBoost | **0.92** | **0.81** | **0.81** | **0.81** |

For the tuned XGBoost model, the notebook identifies **258 false positives** and **256 false negatives** on 6,305 test cases. It also fits a sigmoid-calibrated version of the selected model and plots calibration curves. SHAP analysis shows overall feature contributions and explains one applicant's score; these are model explanations, not proof of causation.

## Scope and current limitations

The notebook's later threshold experiment uses `precisions - recalls` in place of `precisions * recalls` in the F1 formula. It selects a threshold shown as **1.00**, producing zero positive predictions in that experiment. The saved `best_threshold.pkl` and the API's high/low risk decision therefore need correction before the API classification should be relied upon. The **0.92 accuracy and 0.81 F1 above belong to the tuned XGBoost model at its default threshold**, not to that saved threshold or the calibrated API decision. The API loads the calibrated model and threshold, while the notebook's SHAP plots explain a separately fitted, uncalibrated XGBoost model.

## Files

| File | Purpose |
| --- | --- |
| `Credit_Risk.ipynb` | Data analysis, modeling, evaluation, calibration, and SHAP explanations |
| `credit_risk_dataset.csv` | Project dataset |
| `credit_risk_model.pkl` | Saved calibrated model |
| `best_threshold.pkl` | Saved experimental threshold; correction needed as noted above |
| `main.py` | FastAPI prediction endpoint |
