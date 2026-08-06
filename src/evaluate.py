import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(y_true, y_pred):
    """
    Computes and displays classification metrics.
    """

    print("\n================ MODEL EVALUATION REPORT ================")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Legitimate (0)", "Fraudulent (1)"],
        )
    )