from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# PATHS


DATA_PATH = Path("data/model_data.csv")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


# FEATURES


categorical_features = [
    "type",
]

numeric_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "originBalanceChange",
    "destBalanceChange",
    "originBalanceError",
    "destBalanceError",
    "originEmptied",
]

model_features = (
    categorical_features
    + numeric_features
)


# LOAD MODELING DATA


df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print(f"Records loaded: {len(df):,}")



# CHRONOLOGICAL TRAIN / TEST SPLIT


cutoff_step = df["step"].quantile(0.80)

train_df = df[
    df["step"] <= cutoff_step
].copy()

test_df = df[
    df["step"] > cutoff_step
].copy()

print(f"Training records: {len(train_df):,}")
print(f"Testing records: {len(test_df):,}")


# PREPARE FEATURES


X_train = train_df[
    model_features
]

X_test = test_df[
    model_features
]

y_test = test_df[
    "isFraud"
]


# PREPROCESSING


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape: "
    f"{X_test_processed.shape}"
)


# TRAIN ISOLATION FOREST


model = IsolationForest(
    n_estimators=100,
    contamination=0.003,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Isolation Forest...")

model.fit(
    X_train_processed
)

print("Training complete.")



# SAVE MODEL

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

print("\nModel saved:")
print(MODEL_PATH)

print("\nPreprocessor saved:")
print(PREPROCESSOR_PATH)


# SAVE TEST DATA FOR EVALUATION


test_output = test_df[
    [
        "step",
        "type",
        "amount",
        "isFraud",
    ]
].copy()

predictions = model.predict(
    X_test_processed
)

test_output["prediction"] = (
    predictions == -1
).astype(int)

test_output["anomaly_score"] = (
    -model.decision_function(
        X_test_processed
    )
)

test_output.to_csv(
    MODEL_DIR / "test_results.csv",
    index=False,
)

print("\nTest results saved:")
print(
    MODEL_DIR / "test_results.csv"
)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
