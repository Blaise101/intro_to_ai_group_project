import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(y_true, y_pred):
    """
    Computes and displays classification metrics focused on Precision, Recall, and F1-Score.
    """
    print("\n================ MODEL EVALUATION REPORT ================")
    print(classification_report(y_true, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)']))
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}  | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")
    print("=========================================================\n")
    
    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}