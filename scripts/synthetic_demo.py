import torch
from models.rceanet import RCEANet


def run_demo():
    model = RCEANet(backbone_name="efficientnet_b3", num_classes=4)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(dummy_input)

    print("Probabilities:")
    print(outputs["probs"])

    print("Uncertainty:")
    print(outputs["uncertainty"])

    print("Reliability:")
    print(outputs["reliability"])


if __name__ == "__main__":
    run_demo()