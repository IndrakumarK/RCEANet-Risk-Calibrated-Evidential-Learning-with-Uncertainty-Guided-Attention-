import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Dirichlet Evidential Classification Head
    """

    def __init__(self, in_features, num_classes):
        super(EvidentialHead, self).__init__()

        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): pooled features [N, D]

        Returns:
            alpha (torch.Tensor): Dirichlet parameters [N, K]
            probs (torch.Tensor): expected probabilities [N, K]
            uncertainty (torch.Tensor): epistemic uncertainty [N, 1]
        """

        logits = self.fc(x)

        # Ensure non-negative evidence
        evidence = F.softplus(logits)

        # Dirichlet parameters
        alpha = evidence + 1.0

        S = torch.sum(alpha, dim=1, keepdim=True)

        probs = alpha / S

        # Epistemic uncertainty
        K = self.num_classes
        uncertainty = K / S

        return alpha, probs, uncertainty