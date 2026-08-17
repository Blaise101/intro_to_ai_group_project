from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# PAGE CONFIGURATION
st.set_page_config(
    page_title="MoMo Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)


# FILE PATHS
MODEL_PATH = Path("models/isolation_forest.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
RESULTS_PATH = Path("models/test_results.csv")


# LOAD SAVED MODEL
@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


@st.cache_data
def load_results():

    return pd.read_csv(RESULTS_PATH)


try:
    model, preprocessor = load_model()
    results = load_results()

except FileNotFoundError:
    st.error("Required model files were not found. Run python src/model.py first.")

    st.stop()


# SELECTED ANOMALY THRESHOLD
SELECTED_RATE = 0.01


score_threshold = results["anomaly_score"].quantile(1 - SELECTED_RATE)


results["IsolationForestResult"] = (results["anomaly_score"] >= score_threshold).astype(
    int
)


# COMPARE ISOLATION FOREST WITH PAYSIM LABEL
def comparison_label(row):

    prediction = row["IsolationForestResult"]
    actual = row["isFraud"]

    if prediction == 1 and actual == 1:
        return "Correct Fraud Alert"

    if prediction == 1 and actual == 0:
        return "False Alert"

    if prediction == 0 and actual == 1:
        return "Missed Fraud"

    return "Correct Normal"


results["Comparison"] = results.apply(
    comparison_label,
    axis=1,
)

# SUMMARY VALUES
total_transactions = len(results)

actual_fraud = int(results["isFraud"].sum())

flagged_transactions = int(results["IsolationForestResult"].sum())

correct_fraud_alerts = int((results["Comparison"] == "Correct Fraud Alert").sum())

false_alerts = int((results["Comparison"] == "False Alert").sum())

missed_fraud = int((results["Comparison"] == "Missed Fraud").sum())

correct_normal = int((results["Comparison"] == "Correct Normal").sum())


# PERFORMANCE METRICS
precision = (
    correct_fraud_alerts / flagged_transactions if flagged_transactions > 0 else 0
)

recall = correct_fraud_alerts / actual_fraud if actual_fraud > 0 else 0

f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


# CIRCULAR PERFORMANCE CHART
def performance_donut(value, title):

    percentage = value * 100

    figure = go.Figure(
        data=[
            go.Pie(
                values=[
                    percentage,
                    100 - percentage,
                ],
                hole=0.72,
                sort=False,
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )

    figure.add_annotation(
        text=(
            f"<b>{percentage:.2f}%</b><br><span style='font-size:14px'>{title}</span>"
        ),
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=23),
    )

    figure.update_layout(
        height=270,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    return figure


# HEADER
st.title("🛡️ MoMo Fraud Monitoring System")

st.caption("Unsupervised transaction anomaly detection using Isolation Forest")


# MAIN KPI CARDS
col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Transactions Evaluated",
    f"{total_transactions:,}",
)


col2.metric(
    "Known Fraud",
    f"{actual_fraud:,}",
)


col3.metric(
    "Flagged Suspicious",
    f"{flagged_transactions:,}",
)


col4.metric(
    "Correct Fraud Alerts",
    f"{correct_fraud_alerts:,}",
)


st.divider()


# MODEL OUTCOME
st.subheader("Model Outcome")


left, right = st.columns([1.4, 1])


# FRAUD OUTCOME BAR CHART
with left:
    fraud_outcome_df = pd.DataFrame(
        {
            "Outcome": [
                "Correct Fraud Alert",
                "False Alert",
                "Missed Fraud",
            ],
            "Transactions": [
                correct_fraud_alerts,
                false_alerts,
                missed_fraud,
            ],
        }
    )

    figure = px.bar(
        fraud_outcome_df,
        x="Outcome",
        y="Transactions",
        text="Transactions",
        title="Fraud Detection Outcomes",
    )

    figure.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    figure.update_layout(
        xaxis_title="Outcome",
        yaxis_title="Number of Transactions",
        showlegend=False,
        height=430,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


# EVALUATION SUMMARY TABLE
with right:
    st.markdown("### Evaluation Summary")

    summary_table = pd.DataFrame(
        {
            "Outcome": [
                "Correct Fraud Alert",
                "False Alert",
                "Missed Fraud",
                "Correct Normal",
            ],
            "Transactions": [
                correct_fraud_alerts,
                false_alerts,
                missed_fraud,
                correct_normal,
            ],
        }
    )

    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Correct Normal is kept in the table but excluded "
        "from the chart because its much larger value would "
        "make the fraud-related outcomes difficult to compare."
    )


st.divider()


# THRESHOLD PERFORMANCE
st.subheader("Threshold Performance")

st.caption("Performance of the selected 1% anomaly threshold.")


precision_col, recall_col, f1_col = st.columns(3)


with precision_col:
    st.plotly_chart(
        performance_donut(
            precision,
            "Precision",
        ),
        use_container_width=True,
    )


with recall_col:
    st.plotly_chart(
        performance_donut(
            recall,
            "Recall",
        ),
        use_container_width=True,
    )


with f1_col:
    st.plotly_chart(
        performance_donut(
            f1,
            "F1-score",
        ),
        use_container_width=True,
    )


st.success(
    "The 1% anomaly threshold produced the highest "
    "F1-score among the tested thresholds."
)


# THRESHOLD COMPARISON
threshold_rates = [
    0.003,
    0.005,
    0.010,
    0.015,
    0.020,
]


threshold_rows = []


for rate in threshold_rates:
    threshold = results["anomaly_score"].quantile(1 - rate)

    prediction = (results["anomaly_score"] >= threshold).astype(int)

    tp = int(((prediction == 1) & (results["isFraud"] == 1)).sum())

    fp = int(((prediction == 1) & (results["isFraud"] == 0)).sum())

    fn = int(((prediction == 0) & (results["isFraud"] == 1)).sum())

    p = tp / (tp + fp) if tp + fp > 0 else 0

    r = tp / (tp + fn) if tp + fn > 0 else 0

    threshold_f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    threshold_rows.append(
        {
            "Threshold": f"{rate:.1%}",
            "Precision": f"{p:.2%}",
            "Recall": f"{r:.2%}",
            "F1-score": f"{threshold_f1:.2%}",
        }
    )


threshold_table = pd.DataFrame(threshold_rows)


st.dataframe(
    threshold_table,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# SUSPICIOUS TRANSACTIONS
st.subheader("Suspicious Transactions")


suspicious_df = (
    results[results["IsolationForestResult"] == 1]
    .sort_values(
        "anomaly_score",
        ascending=False,
    )
    .copy()
)


st.write(
    f"Isolation Forest flagged "
    f"**{len(suspicious_df):,} transactions** "
    f"for further review."
)


st.dataframe(
    suspicious_df[
        [
            "step",
            "type",
            "amount",
            "anomaly_score",
            "isFraud",
            "Comparison",
        ]
    ].head(100),
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ANALYZE NEW TRANSACTION
st.subheader("Analyze a New Transaction")

st.caption(
    "Enter transaction information to determine whether "
    "the transaction appears unusual to the trained model."
)


with st.form("transaction_form"):
    left, right = st.columns(2)

    with left:
        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "TRANSFER",
                "CASH_OUT",
            ],
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=10000.0,
        )

        old_origin = st.number_input(
            "Origin Balance Before",
            min_value=0.0,
            value=20000.0,
        )

    with right:
        new_origin = st.number_input(
            "Origin Balance After",
            min_value=0.0,
            value=10000.0,
        )

        old_dest = st.number_input(
            "Destination Balance Before",
            min_value=0.0,
            value=5000.0,
        )

        new_dest = st.number_input(
            "Destination Balance After",
            min_value=0.0,
            value=15000.0,
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
    )


# NEW TRANSACTION PREDICTION
if submitted:
    # FEATURE ENGINEERING
    origin_change = old_origin - new_origin

    dest_change = new_dest - old_dest

    origin_error = abs((old_origin - amount) - new_origin)

    dest_error = abs((old_dest + amount) - new_dest)

    origin_emptied = int(new_origin == 0)

    # CREATE MODEL INPUT
    transaction = pd.DataFrame(
        {
            "type": [transaction_type],
            "amount": [amount],
            "oldbalanceOrg": [old_origin],
            "newbalanceOrig": [new_origin],
            "oldbalanceDest": [old_dest],
            "newbalanceDest": [new_dest],
            "originBalanceChange": [origin_change],
            "destBalanceChange": [dest_change],
            "originBalanceError": [origin_error],
            "destBalanceError": [dest_error],
            "originEmptied": [origin_emptied],
        }
    )

    # PREPROCESS
    processed = preprocessor.transform(transaction)

    # ANOMALY SCORE
    anomaly_score = float(-model.decision_function(processed)[0])

    suspicious = anomaly_score >= score_threshold

    # RESULT
    st.markdown("### Analysis Result")

    result1, result2 = st.columns(2)

    result1.metric(
        "Anomaly Score",
        f"{anomaly_score:.4f}",
    )

    result2.metric(
        "Model Assessment",
        ("Suspicious" if suspicious else "Normal"),
    )

    # RECOMMENDATION
    if suspicious:
        st.warning(
            "Recommendation: This transaction shows unusual "
            "behaviour. Perform additional verification and "
            "manual review before taking action."
        )

    else:
        st.success(
            "Recommendation: This transaction does not exceed "
            "the selected anomaly threshold. Continue normal "
            "monitoring."
        )

    # TRANSACTION DETAILS
    details = pd.DataFrame(
        {
            "Indicator": [
                "Transaction Type",
                "Amount",
                "Origin Balance Change",
                "Destination Balance Change",
                "Origin Balance Error",
                "Destination Balance Error",
                "Origin Account Emptied",
            ],
            "Value": [
                transaction_type,
                f"{amount:,.2f}",
                f"{origin_change:,.2f}",
                f"{dest_change:,.2f}",
                f"{origin_error:,.2f}",
                f"{dest_error:,.2f}",
                ("Yes" if origin_emptied else "No"),
            ],
        }
    )

    st.markdown("### Transaction Indicators")

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

# MODEL INFORMATION
st.divider()


st.info(
    """
    Isolation Forest was trained without using the isFraud label.
    PaySim fraud labels are used only after prediction to evaluate
    how well the detected anomalies correspond to known fraud.

    A suspicious transaction is not automatically proof of fraud.
    Flagged transactions should be reviewed or verified.
    """
)
