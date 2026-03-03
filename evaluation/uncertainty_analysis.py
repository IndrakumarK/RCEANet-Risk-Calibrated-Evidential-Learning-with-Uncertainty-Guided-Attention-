import torch
import numpy as np


def compute_uncertainty_error_correlation(probs, labels, uncertainty):
    """
    Computes Pearson correlation between predictive uncertainty and error indicator.

    Args:
        probs (torch.Tensor): predicted probabilities [N, K]
        labels (torch.Tensor): ground truth labels [N]
        uncertainty (torch.Tensor): uncertainty values [N] or [N,1]

    Returns:
        float: correlation coefficient
    """

    # Move to CPU numpy
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()
    uncertainty = uncertainty.detach().cpu().view(-1)

    predictions = torch.argmax(probs, dim=1)

    error_indicator = (predictions != labels).float()

    uncertainty_np = uncertainty.numpy()
    error_np = error_indicator.numpy()

    # Handle constant vectors
    if np.std(uncertainty_np) == 0 or np.std(error_np) == 0:
        return 0.0

    correlation = np.corrcoef(uncertainty_np, error_np)[0, 1]

    return float(correlation)