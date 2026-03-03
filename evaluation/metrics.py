import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def compute_classification_metrics(probs, labels, num_classes):
    """
    Computes classification metrics for multi-class setting.

    Args:
        probs (torch.Tensor): [N, K] predicted probabilities
        labels (torch.Tensor): [N] ground truth labels
        num_classes (int)

    Returns:
        dict of metrics
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    preds = np.argmax(probs_np, axis=1)

    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(labels_np, preds)

    # Macro metrics
    metrics["precision_macro"] = precision_score(
        labels_np, preds, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        labels_np, preds, average="macro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        labels_np, preds, average="macro", zero_division=0
    )

    # AUC (One-vs-Rest)
    try:
        metrics["auc_ovr"] = roc_auc_score(
            labels_np,
            probs_np,
            multi_class="ovr"
        )
    except ValueError:
        metrics["auc_ovr"] = np.nan

    return metrics