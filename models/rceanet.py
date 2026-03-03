import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from models.attention import UncertaintyGuidedAttention
from models.evidential_head import EvidentialHead


class RCEANet(nn.Module):
    """
    Risk-Calibrated Evidential Attention Network
    """

    def __init__(self, backbone_name="efficientnet_b3",
                 num_classes=4,
                 pretrained=True):

        super(RCEANet, self).__init__()

        # Backbone
        self.backbone = get_backbone(backbone_name, pretrained)
        feature_dim = self.backbone.out_channels

        # Attention
        self.attention = UncertaintyGuidedAttention(feature_dim)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Evidential head
        self.evidential_head = EvidentialHead(
            in_features=feature_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Returns:
            alpha
            probs
            uncertainty
            reliability
            attention_map
        """

        # Feature extraction
        features = self.backbone(x)  # [N, C, H, W]

        # Global pooling (initial)
        pooled = self.pool(features).view(features.size(0), -1)

        # Initial evidential prediction
        alpha, probs, uncertainty = self.evidential_head(pooled)

        # Apply uncertainty-guided attention
        refined_features, attention_map = self.attention(
            features,
            uncertainty
        )

        # Pool refined features
        pooled_refined = self.pool(refined_features).view(
            refined_features.size(0), -1
        )

        # Final evidential prediction (after attention refinement)
        alpha, probs, uncertainty = self.evidential_head(pooled_refined)

        # Reliability computation
        max_prob, _ = torch.max(probs, dim=1, keepdim=True)
        reliability = (1.0 - uncertainty) * max_prob

        return {
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "attention_map": attention_map
        }