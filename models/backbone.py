import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()

        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        # Remove classifier
        self.features = model.features
        self.out_channels = model.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)
        return x


class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtBackbone, self).__init__()

        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        # Extract feature extractor
        self.features = model.features
        self.out_channels = model.classifier[2].in_features

    def forward(self, x):
        x = self.features(x)
        return x


def get_backbone(name="efficientnet_b3", pretrained=True):
    if name == "efficientnet_b3":
        backbone = EfficientNetBackbone(pretrained)
    elif name == "convnext_tiny":
        backbone = ConvNeXtBackbone(pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return backbone