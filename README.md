# 🛡️ Mobile Money (MoMo) Anomaly Fraud Detector

---

## 📌 Project Overview

This project uses **Isolation Forest**, an **unsupervised machine-learning algorithm**, to identify unusual transaction behaviour in mobile-money transactions.

The model learns transaction patterns without using the `isFraud` label during training. The known fraud labels are used after prediction to evaluate how well the detected anomalies match known fraud cases.

The system is designed as a **transaction monitoring and screening tool**. A flagged transaction is considered unusual by the model and does not automatically mean that it is fraudulent.

### Key Features

* **Anomaly Scoring:** Assigns anomaly scores to transactions based on learned transaction behaviour.
* **Threshold Evaluation:** Compares different anomaly-rate thresholds to identify a suitable operating point.
* **Transaction Filtering:** Allows users to inspect flagged transactions.
* **Visualizations:** Displays transaction and anomaly results through charts.
* **Interactive Dashboard:** Provides a Streamlit interface for viewing and demonstrating the model.

---

## 👥 Team Roles

| Team Member | Main Responsibility |
| :--- | :--- |
| **Remy** | Model development — `model.py` |
| **Egide** | Data loading and preparation — `data_loader.py` |
| **Blaise** | Model evaluation — `evaluate.py` |
| **Fida** | Streamlit application — `app.py` |

---

## 📊 Dataset

The project uses the **PaySim mobile-money transaction dataset**.

- Original transactions: **6,362,620**
- Known fraud transactions: **8,213**
- Final modeling transactions: **2,770,409**
- Test transactions: **552,504**
- Known fraud in test set: **4,258**

The final modeling dataset contains **CASH_OUT and TRANSFER** transactions.

---

## 📈 Model Results

Five anomaly-rate thresholds were evaluated.

The **1% threshold produced the highest F1-score among the tested thresholds**.

| Metric | Result |
| :--- | ---: |
| Precision | 25.04% |
| Recall | 32.50% |
| F1-score | 28.29% |
| Transactions flagged | 5,527 |
| Correct fraud alerts | 1,384 |
| False alerts | 4,143 |
| Missed fraud | 2,874 |

---

## 📁 Project Structure

```text
.
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

---

## 🚀 How to Run the Project

### 1. Open the project folder

```powershell
cd "C:\Users\kagid\Desktop\Intro To AI Project"
```

### 2. Create a Python virtual environment

This project uses a **Python virtual environment**, not a virtual machine.

```powershell
python -m venv aiprojectenv
```

### 3. Activate the virtual environment

```powershell
.\aiprojectenv\Scripts\Activate.ps1
```

If PowerShell reports an execution-policy error, run:

```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

Then activate the environment again:

```powershell
.\aiprojectenv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open the URL in a browser to use the dashboard.

---

## 🧪 Run the Model and Evaluation

To train the model:

```powershell
python src/model.py
```

To evaluate the model:

```powershell
python src/evaluate.py
```

The trained model and preprocessing pipeline are stored in:

```text
models/
```

---

## ⚠️ Important Note

The Isolation Forest model does **not** use `isFraud` during training.

The `isFraud` label is used only after prediction to compare the model's anomaly results with known fraud labels.

Therefore:

> **Anomaly does not automatically mean fraud.**
