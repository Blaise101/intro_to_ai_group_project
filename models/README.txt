# MoMo Fraud Detection Using Isolation Forest

## Project Overview

This project develops an unsupervised machine learning system for identifying suspicious mobile-money transactions.

The project uses the PaySim synthetic mobile-money transaction dataset and the Isolation Forest anomaly-detection algorithm.

## Problem

Mobile money systems process millions of transactions, while fraudulent activity represents only a very small proportion of those transactions.

Traditional rule-based systems may fail to identify new or unusual fraud patterns.

This project investigates whether unsupervised anomaly detection can identify unusual transaction behaviour that may require investigation.

## Project Workflow

1. Data cleaning
2. Exploratory Data Analysis
3. Transaction-type analysis
4. Modeling-data selection
5. Feature engineering
6. Isolation Forest training
7. Model evaluation
8. Streamlit dashboard

## Modeling Population

EDA showed that known fraud in PaySim occurs in:

- TRANSFER
- CASH_OUT

The anomaly detector therefore focuses on these transaction types.

## Model Features

The model uses:

- Transaction type
- Amount
- Origin balance before transaction
- Origin balance after transaction
- Destination balance before transaction
- Destination balance after transaction
- Origin balance change
- Destination balance change
- Origin balance error
- Destination balance error
- Whether the origin account became empty

The `isFraud` variable is not used as an Isolation Forest training feature.

It is retained only for model evaluation.

## Algorithm

Isolation Forest

Isolation Forest identifies unusual observations by isolating transactions through randomized decision trees.

## Evaluation

The system is evaluated using:

- Precision
- Recall
- F1-score
- Average Precision
- Confusion Matrix
- False Positives
- False Negatives

## Dashboard

The Streamlit application provides:

- Model performance summary
- Fraud detection charts
- Suspicious transaction queue
- Anomaly scores
- New transaction analysis
- Risk level
- Recommended action

## Running the Project

### 1. Install packages

```bash
pip install -r requirements.txt