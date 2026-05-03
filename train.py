"""
ResNet18 ile çocuk çizimlerinden psikolojik skor tahmini.

Transfer learning:
  - ImageNet ağırlıkları yüklenir
  - İlk katmanlar dondurulur, son 2 blok + FC ince ayar yapılır

Cikti: happiness, sadness, anger, fear (0-1) — 4 duygu skoru
Wellbeing: tahmin sonrası dinamik formülle hesaplanır (modelin çıktısı değil)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import DrawingDataset, SCORE_COLS
import numpy as np
import pandas as pd

# ── Ayarlar ───────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
PATIENCE   = 6
DEVICE     = torch.device("cuda")

TRAIN_CSV  = r"c:\ML_Project\train_labels.csv"
TEST_CSV   = r"c:\ML_Project\test_labels.csv"
MODEL_PATH = r"c:\ML_Project\model_resnet.pt"

# ── Veri yükleyiciler ─────────────────────────────────────────
train_ds = DrawingDataset(TRAIN_CSV, train=True)
test_ds  = DrawingDataset(TEST_CSV,  train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}")

# ── Model ─────────────────────────────────────────────────────
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for name, param in resnet.named_parameters():
    if not any(name.startswith(x) for x in ("layer3", "layer4", "fc")):
        param.requires_grad = False

resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 4),
    nn.Sigmoid(),
)

resnet = resnet.to(DEVICE)
trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Egitilebilir parametre: {trainable:,}\n")

# ── Loss ve optimizer ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, resnet.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

# ── Eğitim döngüsü ────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0

print(f"{'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>10}")
print("-" * 46)

for epoch in range(1, EPOCHS + 1):

    resnet.train()
    train_losses = []
    for imgs, scores in train_loader:
        imgs   = imgs.to(DEVICE)
        scores = scores.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(resnet(imgs), scores)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    resnet.eval()
    val_losses, val_maes = [], []
    with torch.no_grad():
        for imgs, scores in test_loader:
            imgs   = imgs.to(DEVICE)
            scores = scores.to(DEVICE)
            preds  = resnet(imgs)
            val_losses.append(criterion(preds, scores).item())
            val_maes.append((preds - scores).abs().mean().item())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    val_mae    = np.mean(val_maes)

    scheduler.step(val_loss)
    print(f"{epoch:<8} {train_loss:>12.5f} {val_loss:>12.5f} {val_mae:>10.4f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(resnet.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping - epoch {epoch}")
            break

# ── En iyi modeli yükle ───────────────────────────────────────
print(f"\nEn iyi val loss: {best_val_loss:.5f}")
resnet.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
resnet.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, scores in test_loader:
        all_preds.append(resnet(imgs.to(DEVICE)).cpu())
        all_labels.append(scores)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print(f"\n{'Skor':<28} {'MAE':>8}")
print("-" * 38)
for i, col in enumerate(SCORE_COLS):
    mae = np.abs(all_preds[:, i] - all_labels[:, i]).mean()
    print(f"{col:<28} {mae:>8.4f}")

# ── Wellbeing: dinamik ağırlıklı formül ──────────────────────
def compute_wellbeing(h, s, a, f):
    neg_weight = 1.0 - h * 0.6
    wb = (
        h * 0.55
        + (1 - s * neg_weight) * 0.20
        + (1 - a * neg_weight) * 0.15
        + (1 - f * neg_weight) * 0.10
    )
    return float(np.clip(wb, 0, 1) * 100)

# ── Test tahminlerini CSV/Excel'e kaydet ──────────────────────
test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
pred_df = test_df[["file_path", "filename", "label"]].copy()

for i, col in enumerate(SCORE_COLS):
    pred_df[f"pred_{col}"] = np.round(all_preds[:, i], 4)

pred_df["pred_psychological_wellbeing"] = [
    round(compute_wellbeing(
        all_preds[i, 0], all_preds[i, 1],
        all_preds[i, 2], all_preds[i, 3]
    ), 1)
    for i in range(len(all_preds))
]

pred_df.to_csv(
    r"c:\ML_Project\test_predictions.csv", index=False, encoding="utf-8-sig"
)
pred_df.to_excel(
    r"c:\ML_Project\test_predictions.xlsx", index=False, engine="openpyxl"
)

wb_mean = pred_df["pred_psychological_wellbeing"].mean()
print(f"\nOrtalama wellbeing (test): {wb_mean:.1f} / 100")
print(f"Tahminler kaydedildi: test_predictions.csv / .xlsx")
print(f"Model kaydedildi: {MODEL_PATH}")
