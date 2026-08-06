# 🛡️ Mobile Money (MoMo) Anomaly Fraud Detector# 🛡️ Mobile Money (MoMo) Anomaly Fraud Detector

---

## 📌 Project Overview

Traditional rule-based fraud detection systems often struggle to catch novel or evolving financial fraud patterns. This project implements an **Unsupervised Learning model** (Outlier / Anomaly Detection) that identifies unusual transaction behavior without requiring predefined fraud labels.

### Key Features:
* **Real-Time Anomaly Scoring:** Evaluates dataset transactions and flags potential anomalies based on continuous risk scores.
* **Dynamic Sensitivity Tuning:** Interactive sidebar slider to adjust the model contamination threshold on the fly.
* **Interactive Data Filtering:** One-click toggle to isolate and inspect flagged high-risk transactions.
* **Distribution Visualizations:** Scatter plots mapping transaction amounts against anomaly scores to visualize outliers.

---

## 👥 Team Roles & Presentation Breakdown

To ensure seamless coordination during the project presentation, responsibilities are divided into four core domain areas:

| Role | Focus Area | Key AI Concepts Covered |
| :--- | :--- | :--- |
| **1. Data Preprocessing Specialist** | Feature Engineering & Data Structure | Rule-Based vs ML, Feature Scaling, Data Vectors |
| **2. AI Model Architect** | Core Algorithm & Pattern Recognition | Unsupervised Learning, Distance/Partitioning Outliers |
| **3. Model Evaluation Lead** | Threshold Tuning & Metric Analysis | Anomaly Scoring, Contamination Hyperparameters, Class Imbalance |
| **4. Deployment & UX Lead** | Streamlit Dashboard & Human-in-the-Loop AI | AI System Deployment, Decision Support Systems, Visual Interpretability |

---

### Development

#### 1. Directories

```bash
    momo-fraud-detector/
    │
    ├── data/
    │   └── dataset.csv             # Kaggle PaySim / Credit Card sample dataset
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py          # Data ingestion & scaling pipeline
    │   ├── model.py                # Isolation Forest training & inference logic
    │   └── evaluate.py             # Metrics calculation & confusion matrix generator
    │
    ├── app.py                      # Interactive Streamlit dashboard
    ├── requirements.txt            # Python dependencies
    └── README.md                   # Setup guide and usage instructions
```

#### 1. Set Up a Virtual Environment

```bash

python -m venv aiprojectenv
.\aiprojectenv\Scripts\Activate.ps1

_(Note: If PowerShell throws a script execution policy error, run `Set-ExecutionPolicy Unrestricted -Scope Process` first, then activate)._
```
#### 2. Install Required Dependencies
