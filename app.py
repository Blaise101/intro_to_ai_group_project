import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_and_preprocess_data
from src.model import FraudDetector

st.set_page_config(page_title="MoMo Fraud Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Mobile Money (MoMo) Anomaly Fraud Detector")
st.markdown("Unsupervised ML system identifying high-risk transactions in real-time.")

st.sidebar.header("Model Configuration")
contamination = st.sidebar.slider(
    "Contamination Threshold (Sensitivity)", 0.0001, 0.01, 0.001, step=0.0005
)


@st.cache_data
def run_pipeline(contam_val):
    X_train, X_test, _y_train, y_test, _preprocessor = load_and_preprocess_data(
        "data/dataset.csv"
    )
    detector = FraudDetector(contamination=contam_val)
    detector.train(X_train)
    preds = detector.predict(X_test)
    scores = detector.get_anomaly_scores(X_test)
    return X_test, y_test, preds, scores


try:
    X_test, y_test, preds, scores = run_pipeline(contamination)

    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Test Transactions", len(preds))
    col2.metric("Flagged Anomalies", int(np.sum(preds)))
    col3.metric("Normal Transactions", int(len(preds) - np.sum(preds)))

    st.subheader("Transaction Risk Analysis")
    results_df = pd.DataFrame(
        {
            "Anomaly Score": scores,
            "Flagged as Fraud": ["🚨 FRAUD" if p == 1 else "✅ Normal" for p in preds],
        }
    )

    st.dataframe(results_df.head(100), use_container_width=True)

except FileNotFoundError as e:
    st.info(
        "Please place your `dataset.csv` inside the `data/` directory to run the dashboard preview."
    )
    st.error(f"Details: {e}")
