from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Loads transaction data, cleans features, handles encoding & scaling,
    and returns split datasets ready for Isolation Forest.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    print("Loading transaction dataset...")
    df = pd.read_csv(file_path)
    # If dataset has PaySim columns, process them
    if 'type' in df.columns:
        # Drop non-informative string columns
        drop_cols = [c for c in ['nameOrig', 'nameDest', 'isFlaggedFraud'] if c in df.columns]
        df = df.drop(columns=drop_cols)
        
        # Separate target label for evaluation
        y = df['isFraud'] if 'isFraud' in df.columns else None
        X = df.drop(columns=['isFraud']) if 'isFraud' in df.columns else df.copy()

        # Identify categorical and numeric columns
        categorical_cols = ['type']
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        # Preprocessing pipeline: One-Hot Encode categorical, Scale numerical
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ]
        )
        
        X_processed = preprocessor.fit_transform(X)
        
    else:
        # Fallback for standard numerical PCA datasets (like Kaggle Credit Card)
        y = df['Class'] if 'Class' in df.columns else None
        X = df.drop(columns=['Class']) if 'Class' in df.columns else df.copy()
        
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        preprocessor = scaler

    # Train/Test Split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Data preprocessed successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, preprocessor
    else:
        return X_processed, preprocessor

if __name__ == "__main__":
    # Quick standalone test
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("data/dataset.csv")