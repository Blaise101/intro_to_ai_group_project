import numpy as np
from sklearn.ensemble import IsolationForest


class FraudDetector:
    def __init__(self, contamination=0.001, n_estimators=100, random_state=42):
        """
        Initializes Isolation Forest model.
        contamination: expected fraction of anomalies in data.
        """
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train):
        """Trains the Isolation Forest model on unlabelled or normal data."""
        print(f"Training Isolation Forest (contamination={self.contamination})...")
        self.model.fit(X_train)
        print("Training complete.")

    def predict(self, X):
        """
        Predicts whether points are anomalies.
        IsolationForest returns -1 for anomalies and 1 for normal.
        We convert -1 -> 1 (Fraud) and 1 -> 0 (Legitimate).
        """
        raw_preds = self.model.predict(X)
        fraud_preds = np.where(raw_preds == -1, 1, 0)
        return fraud_preds

    def get_anomaly_scores(self, X):
        """Returns raw anomaly scores. Lower scores represent higher anomaly risk."""
        return self.model.score_samples(X)