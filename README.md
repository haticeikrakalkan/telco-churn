\# Telco Customer Churn - ML Pipeline



An end-to-end machine learning project for customer churn prediction:

modular training code, tests, CI, a REST API, Docker, and experiment tracking.



!\[Tests](https://github.com/haticeikrakalkan/telco-churn/actions/workflows/tests.yml/badge.svg)



\## Results



| Metric | Value |

|---|---|

| Accuracy | 0.75 |

| Churn Recall | 0.73 |

| Churn F1 | 0.61 |



Model: AdaBoost + SMOTE (to handle class imbalance), tuned with RandomizedSearchCV.

The goal is to avoid missing customers who are about to leave (high recall), which suits a retention-campaign use case.



\## Architecture



&#x20;   CSV -> clean\_data -> Pipeline (OneHot + Scaler + SMOTE + AdaBoost)

&#x20;       -> model.joblib -> FastAPI /predict -> Docker



Training and the API share the same `clean\_data` function, which prevents training/serving skew.



\## Project Structure



&#x20;   src/data.py       data loading

&#x20;   src/features.py   cleaning and preprocessor

&#x20;   src/train.py      training + MLflow tracking

&#x20;   src/api.py        FastAPI service

&#x20;   tests/            pytest tests

&#x20;   .github/          CI (GitHub Actions)



\## Setup and Usage

