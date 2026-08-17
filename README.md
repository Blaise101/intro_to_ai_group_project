# 🛡️ Mobile Money (MoMo) Anomaly Fraud Detector

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

```C:.
│   .gitignore
│   app.py
│   README.md
│   requirements.txt
│
├───data
│       dataset.csv
│       model_data.csv
│
├───models
│       isolation_forest.pkl
│       preprocessor.pkl
│       test_results.csv
│
├───notebooks
│       eda.ipynb
│
└───src
        data_loader.py
        evaluate.py
        model.py

```

#### 1. Set Up a Virtual Environment

```bash
    python -m venv aiprojectenv
    .\aiprojectenv\Scripts\Activate.ps1
```
_(Note: If PowerShell throws a script execution policy error, run `Set-ExecutionPolicy Unrestricted -Scope Process` first, then activate)._


#### 2. Install Required Dependencies

```bash
    pip install -r requirements.txt
```

#### 2.1  Development
 
  - **Remy:** model.py
  - **Egide:**  data_loader.py
  - **Blaise:**  evaluate.py
  - **Fida:**  app.py


#### 3. Run the application

```bash
    python -m streamlit run app.py
```

**After running the command:**  

    1. The terminal will display your Local URL (typically http://localhost:8501).  
    2. A browser tab will automatically open displaying the interactive dashboard.