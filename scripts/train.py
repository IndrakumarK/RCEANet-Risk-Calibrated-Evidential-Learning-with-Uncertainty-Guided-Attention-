import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =========================
# 🔁 Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# ⚙️ Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 📦 Import your modules
# =========================
from models.rceanet import RCEANet
from utils.dataset import get_dataloaders
from utils.losses import evidential_loss  # your custom loss
from utils.metrics import compute_accuracy

# =========================
# 🔧 Hyperparameters
# =========================
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 4
SAVE_PATH = "best_model.pth"

# =========================
# 📊 Data
# =========================
train_loader, val_loader = get_dataloaders()

# =========================
# 🧠 Model
# =========================
model = RCEANet(num_classes=NUM_CLASSES).to(device)

# =========================
# ⚙️ Optimizer
# =========================
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 📉 Training Function
# =========================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = evidential_loss(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy

# =========================
# 📊 Validation Function
# =========================
def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = evidential_loss(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy

# =========================
# 🚀 Training Loop
# =========================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = validate(model, val_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # =========================
    # 💾 Save Best Model
    # =========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best model saved!")

print("\n🎉 Training Complete!")
