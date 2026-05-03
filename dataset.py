"""
Çocuk çizimi veri seti — PyTorch Dataset sınıfı.
Etiketleri label smoothing ile sürekli skora dönüştürür.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ── Label smoothing tablosu ───────────────────────────────────
SOFT_SCORES = {
    "Happy": [0.85, 0.05, 0.05, 0.05],
    "Sad":   [0.05, 0.82, 0.05, 0.08],
    "Angry": [0.05, 0.10, 0.75, 0.10],
    "Fear":  [0.05, 0.15, 0.05, 0.75],
}
# sıra: happiness, sadness, anger, fear

SCORE_COLS = [
    "happiness_score", "sadness_score",
    "anger_score", "fear_score",
]

# ── Augmentation ──────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def make_transforms(img_size: int = 224):
    train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, val


class DrawingDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True, img_size: int = 224):
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.paths  = df["file_path"].tolist()
        self.labels = df["label"].tolist()
        train_tf, val_tf = make_transforms(img_size)
        self.transform = train_tf if train else val_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        scores = torch.tensor(
            SOFT_SCORES[self.labels[idx]], dtype=torch.float32
        )
        return img, scores
