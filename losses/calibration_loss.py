import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """
    Multi-class Brier-based calibration loss:
    L_calib = sum_k (y_k - p_k)^2
    """

    def __init__(self, num_classes):
        super(CalibrationLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, probs, labels):
        """
        Args:
            probs (torch.Tensor): predicted probabilities [N, K]
            labels (torch.Tensor): ground truth labels [N]

        Returns:
            torch.Tensor: scalar loss
        """

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

        return loss