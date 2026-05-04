"""
Çocuk çizimi veri seti — PyTorch Dataset sınıfı.
Skorlar train_scores.xlsx / test_scores.xlsx dosyasından okunur.
Psikolog skorları değiştirirse bir sonraki eğitimde otomatik yansır.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SOFT_SCORES = {
    "Happy": [0.85, 0.05, 0.05, 0.05],
    "Sad":   [0.05, 0.82, 0.05, 0.08],
    "Angry": [0.05, 0.10, 0.75, 0.10],
    "Fear":  [0.05, 0.15, 0.05, 0.75],
}

SCORE_COLS = [
    "happiness_score", "sadness_score",
    "anger_score", "fear_score",
]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def make_transforms(img_size: int = 224):
    train = transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, val


def _load_scores(xlsx_path: str) -> dict:
    """filename -> [h, s, a, f] skorlarını Excel'den yükler."""
    import pandas as pd
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    score_dict = {}
    for _, row in df.iterrows():
        score_dict[row["filename"]] = [
            float(row["happiness_score"]),
            float(row["sadness_score"]),
            float(row["anger_score"]),
            float(row["fear_score"]),
        ]
    return score_dict


class DrawingDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = True,
                 img_size: int = 224, scores_xlsx: str = None):
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.paths     = df["file_path"].tolist()
        self.filenames = df["filename"].tolist()
        self.labels    = df["label"].tolist()
        train_tf, val_tf = make_transforms(img_size)
        self.transform = train_tf if train else val_tf

        if scores_xlsx and os.path.exists(scores_xlsx):
            self.score_dict = _load_scores(scores_xlsx)
            print(f"Skorlar Excel'den yuklendi: {scores_xlsx}")
        else:
            self.score_dict = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        if self.score_dict is not None:
            scores = self.score_dict[self.filenames[idx]]
        else:
            scores = SOFT_SCORES[self.labels[idx]]

        return img, torch.tensor(scores, dtype=torch.float32)
