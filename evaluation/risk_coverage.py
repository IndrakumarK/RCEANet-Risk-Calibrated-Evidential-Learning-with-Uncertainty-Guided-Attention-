import torch
import numpy as np


def compute_risk_coverage(probs, labels, reliability):
    """
    Computes risk–coverage curve data.

    Args:
        probs (torch.Tensor): predicted probabilities [N, K]
        labels (torch.Tensor): ground truth labels [N]
        reliability (torch.Tensor): reliability scores [N]

    Returns:
        coverage (np.array)
        risk (np.array)
    """

    # Move to CPU numpy
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()
    reliability = reliability.detach().cpu()

    predictions = torch.argmax(probs, dim=1)
    errors = (predictions != labels).float()

    # Sort by reliability (descending)
    sorted_indices = torch.argsort(reliability, descending=True)

    sorted_errors = errors[sorted_indices]

    N = len(labels)

    coverage = []
    risk = []

    cumulative_errors = 0.0

    for i in range(1, N + 1):
        cumulative_errors += sorted_errors[i - 1].item()
        current_risk = cumulative_errors / i
        current_coverage = i / N

        coverage.append(current_coverage)
        risk.append(current_risk)

    return np.array(coverage), np.array(risk)