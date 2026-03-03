import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.rceanet import RCEANet
from data.dataset_loader import BrainTumorDataset
from data.transforms import get_train_transforms, get_test_transforms

from losses.evidential_loss import EvidentialLoss
from losses.kl_regularization import KLDivergenceLoss
from losses.calibration_loss import CalibrationLoss
from losses.attention_alignment_loss import AttentionUncertaintyAlignmentLoss

from evaluation.calibration import compute_ece
from evaluation.metrics import compute_classification_metrics
from utils.seed import set_seed

def train_model(config,
                use_kl=True,
                use_calibration=True,
                use_attention_alignment=True):
set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = BrainTumorDataset(
        root_dir=config["train_dir"],
        transform=get_train_transforms()
    )

    val_dataset = BrainTumorDataset(
        root_dir=config["val_dir"],
        transform=get_test_transforms()
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=4)

    # Model
    model = RCEANet(
        backbone_name=config["backbone"],
        num_classes=config["num_classes"]
    ).to(device)

    # Losses
    edl_loss_fn = EvidentialLoss(config["num_classes"])
    kl_loss_fn = KLDivergenceLoss(config["num_classes"])
    calib_loss_fn = CalibrationLoss(config["num_classes"])
    au_loss_fn = AttentionUncertaintyAlignmentLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    best_val_ece = float("inf")
    best_model_path = "best_model.pth"

    for epoch in range(config["epochs"]):

        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            alpha = outputs["alpha"]
            probs = outputs["probs"]
            uncertainty = outputs["uncertainty"]
            attention_map = outputs["attention_map"]

            # Core losses
            L_edl = edl_loss_fn(alpha, labels)
            L_total = L_edl

            if use_kl:
                L_kl = kl_loss_fn(alpha)
                L_total += config["lambda_kl"] * L_kl

            if use_calibration:
                L_calib = calib_loss_fn(probs, labels)
                L_total += config["lambda_calib"] * L_calib

            if use_attention_alignment:
                L_au = au_loss_fn(attention_map, uncertainty)
                L_total += config["lambda_au"] * L_au

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            total_loss += L_total.item()

        # Validation
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                all_probs.append(outputs["probs"])
                all_labels.append(labels)

        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        val_metrics = compute_classification_metrics(
            probs, labels, config["num_classes"]
        )

        val_ece = compute_ece(probs, labels)

        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val ECE: {val_ece:.4f}")
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_model(config)

import os

os.makedirs("checkpoints", exist_ok=True)
best_model_path = os.path.join("checkpoints", "best_model.pth")

        # Save best model based on calibration
        if val_ece < best_val_ece:
            best_val_ece = val_ece
            torch.save(model.state_dict(), best_model_path)

    return best_model_path