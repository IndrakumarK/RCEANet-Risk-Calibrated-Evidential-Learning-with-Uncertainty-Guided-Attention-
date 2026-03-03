import torch
import torch.nn as nn


class UncertaintyGuidedAttention(nn.Module):
    """
    Spatial Attention Module with Uncertainty Modulation
    """

    def __init__(self, in_channels):
        super(UncertaintyGuidedAttention, self).__init__()

        self.attention_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            bias=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features, uncertainty=None):
        """
        Args:
            features (torch.Tensor): [N, C, H, W]
            uncertainty (torch.Tensor): [N, 1] or None

        Returns:
            refined_features (torch.Tensor): [N, C, H, W]
            attention_map (torch.Tensor): [N, 1, H, W]
        """

        # Compute spatial attention
        attention_map = self.sigmoid(self.attention_conv(features))

        if uncertainty is not None:
            # Ensure correct shape: [N,1,1,1]
            if uncertainty.dim() == 2:
                uncertainty = uncertainty.unsqueeze(-1).unsqueeze(-1)

            reliability = 1.0 - uncertainty
            reliability = reliability.expand_as(attention_map)

            attention_map = attention_map * reliability

        refined_features = features * attention_map

        return refined_features, attention_map