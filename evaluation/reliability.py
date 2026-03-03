import torch


def compute_uncertainty(alpha):
    """
    Computes epistemic uncertainty:
    u(x) = K / S

    Args:
        alpha (torch.Tensor): Dirichlet parameters [N, K]

    Returns:
        torch.Tensor: uncertainty [N, 1]
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = alpha.size(1)
    uncertainty = K / S
    return uncertainty


def compute_reliability(probs, uncertainty):
    """
    Computes reliability score:
    R(x) = (1 - u(x)) * max_k p_k

    Args:
        probs (torch.Tensor): predicted probabilities [N, K]
        uncertainty (torch.Tensor): uncertainty [N, 1]

    Returns:
        torch.Tensor: reliability scores [N, 1]
    """
    max_prob, _ = torch.max(probs, dim=1, keepdim=True)
    reliability = (1 - uncertainty) * max_prob
    return reliability


def selective_prediction(probs, alpha, threshold=0.6):
    """
    Performs reliability-based selective prediction.

    Args:
        probs (torch.Tensor): predicted probabilities [N, K]
        alpha (torch.Tensor): Dirichlet parameters [N, K]
        threshold (float): reliability threshold

    Returns:
        predictions (torch.Tensor)
        reliability (torch.Tensor)
        mask_keep (torch.Tensor): True if prediction retained
    """
    uncertainty = compute_uncertainty(alpha)
    reliability = compute_reliability(probs, uncertainty)

    predictions = torch.argmax(probs, dim=1)

    mask_keep = reliability.squeeze() >= threshold

    return predictions, reliability.squeeze(), mask_keep