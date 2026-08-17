from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


RESULTS_PATH = Path(
    "models/test_results.csv"
)

# LOAD RESULTS


df = pd.read_csv(
    RESULTS_PATH
)

y_true = df[
    "isFraud"
]

scores = df[
    "anomaly_score"
]


print("=" * 70)
print("UNSUPERVISED ANOMALY DETECTION EVALUATION")
print("=" * 70)

print(
    f"Transactions evaluated: "
    f"{len(df):,}"
)

print(
    f"Known fraud labels: "
    f"{int(y_true.sum()):,}"
)



# THRESHOLD COMPARISON


anomaly_rates = [
    0.003,
    0.005,
    0.010,
    0.015,
    0.020,
]

results = []


for rate in anomaly_rates:

    threshold = scores.quantile(
        1 - rate
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append(
        {
            "AnomalyRate": rate,
            "Flagged": int(
                predictions.sum()
            ),
            "CorrectFraudAlerts": tp,
            "FalseAlerts": fp,
            "MissedFraud": fn,
            "CorrectNormal": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


results_df = pd.DataFrame(
    results
)


print("\nTHRESHOLD COMPARISON")
print("-" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "AnomalyRate": "{:.2%}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
        },
    )
)


# USE 1% FOR DEMONSTRATION

selected_rate = 0.01

selected_threshold = scores.quantile(
    1 - selected_rate
)

df[
    "IsolationForestResult"
] = (
    scores >= selected_threshold
).astype(int)


# HUMAN-READABLE COMPARISON

def compare_result(row):

    model_result = row[
        "IsolationForestResult"
    ]

    actual_result = row[
        "isFraud"
    ]

    if (
        model_result == 1
        and actual_result == 1
    ):
        return "Correct Fraud Alert"

    if (
        model_result == 1
        and actual_result == 0
    ):
        return "False Alert"

    if (
        model_result == 0
        and actual_result == 1
    ):
        return "Missed Fraud"

    return "Correct Normal"


df[
    "Comparison"
] = df.apply(
    compare_result,
    axis=1,
)

# COMPARISON SUMMARY

comparison_summary = (
    df[
        "Comparison"
    ]
    .value_counts()
)

print("\n" + "=" * 70)
print("ISOLATION FOREST VS PAYSIM LABEL")
print("=" * 70)

print(
    comparison_summary
)

# SAMPLE TRANSACTIONS

print("\nSAMPLE COMPARISON")
print("-" * 70)

sample_columns = [
    "type",
    "amount",
    "anomaly_score",
    "IsolationForestResult",
    "isFraud",
    "Comparison",
]

print(
    df[
        sample_columns
    ]
    .sample(
        15,
        random_state=42,
    )
    .to_string(
        index=False
    )
)

# FINAL EXPLANATION

print("\n" + "=" * 70)
print("IMPORTANT INTERPRETATION")
print("=" * 70)

print(
    """
Isolation Forest did NOT use isFraud during training.

The model only learned transaction behaviour from the transaction features.

The isFraud column is used here only after prediction to check whether the anomalies found by Isolation Forest match known PaySim fraud labels.

Therefore:

1 = Isolation Forest considers the transaction suspicious.
0 = Isolation Forest considers the transaction normal.

This does not automatically mean fraud or non-fraud.
"""
)