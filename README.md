# **Machine Learning Case Studies**

A collection of end-to-end machine learning projects spanning the full MLOps lifecycle — from raw data and feature engineering through model training, experiment tracking, and production deployment. Each case study is self-contained with its own data pipeline, configuration, and documentation.

## Table of Contents

- 01 · [Purchase Propensity Model](https://github.com/qyang350/Machine-Learning-Case-Study/tree/main/Purchase%20Propensity%20Model)

## Projects 

**01 · Purchase Propensity Model**

Predicts the probability of a customer making a purchase within the next 30 days.

- **Type**: Binary Classification
- **Algorithm**: XGBoost
- **Platform**: Databricks + Unity Catalog
- **Tracking**: MLflow
- **Serving**: Batch Inference (Delta Table)

What it covers:

- Feature engineering pipeline with leakage controls and null imputation
- Automated hyperparameter tuning with Hyperopt (TPE search)
- Model registration and versioning in Unity Catalog Model Registry
- High-throughput batch scoring with model health validation