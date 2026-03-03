import torch
from torch.utils.data import DataLoader

from models.rceanet import RCEANet
from data.dataset_loader import BrainTumorDataset
from data.transforms import get_test_transforms

from evaluation.metrics import compute_classification_metrics
from evaluation.calibration import compute_ece, compute_brier_score, compute_nll
from evaluation.reliability import compute_uncertainty
from evaluation.uncertainty_analysis import compute_uncertainty_error_correlation
from evaluation.risk_coverage import compute_risk_coverage


def evaluate(model_path, data_dir, backbone="efficientnet_b3",
             num_classes=4, batch_size=16, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = BrainTumorDataset(
        root_dir=data_dir,
        transform=get_test_transforms()
    )

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

    # Model
    model = RCEANet(
        backbone_name=backbone,
        num_classes=num_classes
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_alpha = []
    all_uncertainty = []
    all_reliability = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            all_probs.append(outputs["probs"])
            all_alpha.append(outputs["alpha"])
            all_uncertainty.append(outputs["uncertainty"])
            all_reliability.append(outputs["reliability"])
            all_labels.append(labels)

    # Concatenate
    probs = torch.cat(all_probs, dim=0)
    alpha = torch.cat(all_alpha, dim=0)
    uncertainty = torch.cat(all_uncertainty, dim=0)
    reliability = torch.cat(all_reliability, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Classification metrics
    metrics = compute_classification_metrics(
        probs, labels, num_classes
    )

    # Calibration
    ece = compute_ece(probs, labels)
    brier = compute_brier_score(probs, labels, num_classes)
    nll = compute_nll(probs, labels)

    # Uncertainty–error correlation
    corr = compute_uncertainty_error_correlation(
        probs, labels, uncertainty
    )

    # Risk–coverage
    coverage, risk = compute_risk_coverage(
        probs, labels, reliability.squeeze()
    )

    # Print results
    print("=== Classification Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Calibration Metrics ===")
    print(f"ECE: {ece:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"NLL: {nll:.4f}")

    print("\n=== Reliability Metrics ===")
    print(f"Uncertainty–Error Correlation: {corr:.4f}")

    print("\n=== Risk–Coverage ===")
    print(f"Full coverage risk: {risk[-1]:.4f}")

    return {
        "metrics": metrics,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "correlation": corr,
        "coverage": coverage,
        "risk": risk
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet_b3")
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_dir=args.data_dir,
        backbone=args.backbone,
        num_classes=args.num_classes
    )