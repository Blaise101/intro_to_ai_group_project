from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# CONFIGURATION


BASELINE_FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
]

REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "isFraud",
]


# LOAD DATA


def load_dataset(file_path):
    """
    Load the PaySim transaction dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the PaySim CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print("Loading PaySim dataset...")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]:,} rows, "
        f"{df.shape[1]} columns"
    )

    return df



# VALIDATE DATASET STRUCTURE


def validate_dataset(df):
    """
    Check whether the dataset contains the columns required
    by the project.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    return True



# CLEAN DATA


def clean_dataset(df):
    """
    Perform basic PaySim data cleaning.

    The cleaning follows the EDA findings:
    - remove exact duplicate rows
    - remove rows missing required project variables
    - preserve unusual/extreme values because Isolation Forest
      is designed to detect anomalies

    Returns
    -------
    pandas.DataFrame
        Cleaned transaction dataset.
    """

    validate_dataset(df)

    cleaned_df = df.copy()

    original_rows = len(cleaned_df)

    # Remove exact duplicates
    cleaned_df = cleaned_df.drop_duplicates()

    # Remove rows missing required project variables
    cleaned_df = cleaned_df.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Basic validity checks
    invalid_amounts = (
        cleaned_df["amount"] < 0
    ).sum()

    invalid_old_balances = (
        cleaned_df["oldbalanceOrg"] < 0
    ).sum()

    invalid_new_balances = (
        cleaned_df["newbalanceOrig"] < 0
    ).sum()

    if invalid_amounts > 0:
        raise ValueError(
            "Negative transaction amounts were found."
        )

    if invalid_old_balances > 0:
        raise ValueError(
            "Negative old origin balances were found."
        )

    if invalid_new_balances > 0:
        raise ValueError(
            "Negative new origin balances were found."
        )

    print("\nCleaning complete.")
    print(f"Original rows: {original_rows:,}")
    print(f"Clean rows: {len(cleaned_df):,}")
    print(
        f"Rows removed: "
        f"{original_rows - len(cleaned_df):,}"
    )

    return cleaned_df


# PREPARE BASELINE FEATURES


def prepare_baseline_features(df):
    """
    Separate baseline model features from the evaluation label.

    Isolation Forest training features:
        - type
        - amount
        - oldbalanceOrg
        - newbalanceOrig

    isFraud is kept separately and must never be passed
    into Isolation Forest training.

    Returns
    -------
    X : pandas.DataFrame
        Baseline model features.

    y : pandas.Series
        Ground-truth fraud labels used only for evaluation.
    """

    validate_dataset(df)

    X = df[
        BASELINE_FEATURES
    ].copy()

    y = df[
        "isFraud"
    ].copy()

    return X, y


# CREATE PREPROCESSOR

def build_baseline_preprocessor():
    """
    Create the preprocessing pipeline for baseline features.

    Numerical variables are standardized.
    Transaction type is one-hot encoded.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """

    categorical_features = [
        "type"
    ]

    numeric_features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
    ]

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

    return preprocessor


# TRAIN / TEST SPLIT + PREPROCESSING


def load_and_preprocess_data(
    file_path,
    test_size=0.20,
    random_state=42,
):
    """
    Complete baseline loading and preprocessing pipeline.

    Workflow:
        1. load PaySim dataset
        2. clean dataset
        3. separate baseline features and fraud labels
        4. split train/test
        5. fit preprocessing only on training data
        6. transform train and test sets

    Parameters
    ----------
    file_path : str or Path
        Path to PaySim CSV.

    test_size : float, default=0.20
        Fraction of data reserved for testing.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    preprocessor
    cleaned_df
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    # Load
    df = load_dataset(
        file_path
    )

    # Clean
    cleaned_df = clean_dataset(
        df
    )

    # Separate model features and labels
    X, y = prepare_baseline_features(
        cleaned_df
    )

    # Train/test split
    X_train_raw, X_test_raw, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    print("\nTrain / Test Split")
    print("-" * 50)

    print(
        f"Training transactions: "
        f"{len(X_train_raw):,}"
    )

    print(
        f"Testing transactions: "
        f"{len(X_test_raw):,}"
    )

    print(
        f"Fraud in training set: "
        f"{y_train.sum():,}"
    )

    print(
        f"Fraud in testing set: "
        f"{y_test.sum():,}"
    )

    # Build preprocessing pipeline
    preprocessor = build_baseline_preprocessor()

    # Fit ONLY on training data
    X_train = preprocessor.fit_transform(
        X_train_raw
    )

    # Transform test set using fitted preprocessor
    X_test = preprocessor.transform(
        X_test_raw
    )

    # Final quality checks
    if np.isnan(X_train).any():
        raise ValueError(
            "NaN values found in processed training data."
        )

    if np.isnan(X_test).any():
        raise ValueError(
            "NaN values found in processed test data."
        )

    if np.isinf(X_train).any():
        raise ValueError(
            "Infinite values found in processed training data."
        )

    if np.isinf(X_test).any():
        raise ValueError(
            "Infinite values found in processed test data."
        )

    print("\nPreprocessing complete.")
    print(
        f"Processed train shape: "
        f"{X_train.shape}"
    )

    print(
        f"Processed test shape: "
        f"{X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        cleaned_df,
    )



# STANDALONE TEST


if __name__ == "__main__":

    dataset_path = Path(
        "data/dataset.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        cleaned_df,
    ) = load_and_preprocess_data(
        dataset_path
    )

    print("\n" + "=" * 60)
    print("DATA LOADER TEST COMPLETE")
    print("=" * 60)

    print(
        f"Clean dataset rows: "
        f"{len(cleaned_df):,}"
    )

    print(
        f"Training shape: "
        f"{X_train.shape}"
    )

    print(
        f"Testing shape: "
        f"{X_test.shape}"
    )