import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class EvidentialLoss(nn.Module):
    """
    Multi-class Evidential Loss:

    L_EDL = sum_k [(y_k - p_k)^2 + p_k (1 - p_k) / (S + 1)]
    """

    def __init__(self, num_classes):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, alpha, labels):
        """
        Args:
            alpha (torch.Tensor): Dirichlet parameters [N, K]
            labels (torch.Tensor): ground truth labels [N]

        Returns:
            torch.Tensor: scalar loss
        """

        S = torch.sum(alpha, dim=1, keepdim=True)  # [N,1]
        probs = alpha / S  # Expected probability

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        # Squared error term
        squared_error = torch.sum((one_hot - probs) ** 2, dim=1)

        # Variance regularization term
        variance_term = torch.sum(
            probs * (1 - probs) / (S + 1), dim=1
        )

        loss = torch.mean(squared_error + variance_term)

        return loss