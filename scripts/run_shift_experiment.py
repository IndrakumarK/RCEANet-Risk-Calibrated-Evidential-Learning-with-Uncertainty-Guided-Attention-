import torch
import numpy as np
from torch.utils.data import DataLoader

from models.rceanet import RCEANet
from data.dataset_loader import BrainTumorDataset
from data.transforms import get_test_transforms

from evaluation.metrics import compute_classification_metrics
from evaluation.calibration import compute_ece


def apply_gaussian_noise(images, sigma):
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0.0, 1.0)


def apply_intensity_scaling(images, gamma):
    return torch.clamp(images * gamma, 0.0, 1.0)


def run_shift_experiment(model_path,
                         data_dir,
                         backbone="efficientnet_b3",
                         num_classes=4,
                         batch_size=16,
                         device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = BrainTumorDataset(
        root_dir=data_dir,
        transform=get_test_transforms()
    )

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

    model = RCEANet(
        backbone_name=backbone,
        num_classes=num_classes
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    noise_levels = [0.0, 0.05, 0.10, 0.20]

    results = {}

    for sigma in noise_levels:

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                if sigma > 0:
                    images = apply_gaussian_noise(images, sigma)

                outputs = model(images)

                all_probs.append(outputs["probs"])
                all_labels.append(labels)

        probs = torch.cat(all_probs, dim=0)
        labels = torch.cat(all_labels, dim=0)

        metrics = compute_classification_metrics(
            probs, labels, num_classes
        )

        ece = compute_ece(probs, labels)

        results[sigma] = {
            "accuracy": metrics["accuracy"],
            "ece": ece
        }

        print(f"\nSigma = {sigma}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ECE: {ece:.4f}")

    return results