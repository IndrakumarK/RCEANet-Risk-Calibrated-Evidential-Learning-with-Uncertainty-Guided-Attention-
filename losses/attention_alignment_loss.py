import torch
import torch.nn as nn


class AttentionUncertaintyAlignmentLoss(nn.Module):
    """
    L_AU = || A(x) - (1 - u(x)) ||_2^2
    """

    def __init__(self):
        super(AttentionUncertaintyAlignmentLoss, self).__init__()

    def forward(self, attention_map, uncertainty):
        """
        Args:
            attention_map (torch.Tensor): [N, 1, H, W]
            uncertainty (torch.Tensor): [N, 1]

        Returns:
            torch.Tensor: scalar loss
        """

        # Ensure proper shape
        if uncertainty.dim() == 2:
            uncertainty = uncertainty.unsqueeze(-1).unsqueeze(-1)

        # Broadcast (1 - u(x)) spatially
        target = 1.0 - uncertainty

        # Expand to match attention map
        target = target.expand_as(attention_map)

        loss = torch.mean((attention_map - target) ** 2)

        return loss