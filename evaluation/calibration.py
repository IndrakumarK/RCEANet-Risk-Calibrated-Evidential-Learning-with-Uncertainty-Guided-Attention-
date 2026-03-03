import torch
import torch.nn.functional as F
import numpy as np


def compute_ece(probs, labels, n_bins=15):
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    
    Args:
        probs (torch.Tensor): Predicted probabilities [N, K]
        labels (torch.Tensor): Ground-truth labels [N]
        n_bins (int): Number of confidence bins

    Returns:
        float: ECE value
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def compute_brier_score(probs, labels, num_classes):
    """
    Multi-class Brier Score
    """
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()


def compute_nll(probs, labels, eps=1e-12):
    """
    Negative Log-Likelihood
    """
    probs = torch.clamp(probs, min=eps)
    log_probs = torch.log(probs)
    nll = F.nll_loss(log_probs, labels)
    return nll.item()


def reliability_diagram_data(probs, labels, n_bins=15):
    """
    Returns data needed to plot reliability diagram.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)

        if in_bin.sum() > 0:
            bin_acc.append(accuracies[in_bin].float().mean().item())
            bin_conf.append(confidences[in_bin].mean().item())
            bin_counts.append(in_bin.sum().item())
        else:
            bin_acc.append(0)
            bin_conf.append(0)
            bin_counts.append(0)

    return bin_acc, bin_conf, bin_counts