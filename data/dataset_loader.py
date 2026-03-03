import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


class BrainTumorDataset(Dataset):
    """
    Generic Brain Tumor MRI Dataset Loader
    Supports:
    - Predefined train/test folders
    - Automatic label mapping
    - Grayscale to 3-channel conversion
    - Transform injection
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to dataset split (train or test folder)
            transform (callable): Transform pipeline
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        classes = sorted(os.listdir(self.root_dir))
        classes = [c for c in classes if os.path.isdir(os.path.join(self.root_dir, c))]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")  # Ensures 3 channels

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)