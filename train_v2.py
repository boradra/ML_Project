"""
EfficientNet-B2 ile geliştirilmiş model — v2.

İyileştirmeler (v1 ResNet18'e göre):
  1. EfficientNet-B2: ResNet18'den daha güçlü, daha verimli mimari
  2. Tam fine-tuning: tüm katmanlar eğitiliyor
     - Pretrained katmanlar: lr=1e-5 (çok düşük, ağırlıkları bozmadan)
     - Yeni FC katmanı:      lr=1e-3 (yüksek, sıfırdan öğreniyor)
  3. Resim boyutu: 260x260 (EfficientNet-B2 için optimal)

Mevcut model (model_resnet.pt) ve train.py dokunulmaz.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import DrawingDataset, SCORE_COLS
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score

# ── Ayarlar ───────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 40
LR_BACKBONE = 1e-5   # pretrained katmanlar — çok düşük
LR_HEAD     = 1e-3   # yeni FC katmanı — yüksek
PATIENCE   = 8
DEVICE     = torch.device("cuda")
IMG_SIZE   = 260     # EfficientNet-B2 için optimal boyut

TRAIN_CSV  = r"c:\ML_Project\train_labels.csv"
TEST_CSV   = r"c:\ML_Project\test_labels.csv"
MODEL_PATH = r"c:\ML_Project\model_efficientnet.pt"

# ── Veri yükleyiciler ─────────────────────────────────────────
train_ds = DrawingDataset(TRAIN_CSV, train=True,  img_size=IMG_SIZE)
test_ds  = DrawingDataset(TEST_CSV,  train=False, img_size=IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}  |  Boyut: {IMG_SIZE}x{IMG_SIZE}")

# ── Model ─────────────────────────────────────────────────────
efficientnet = models.efficientnet_b2(
    weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
)

# Tüm katmanlar eğitilebilir (tam fine-tuning)
for param in efficientnet.parameters():
    param.requires_grad = True

# FC katmanını 4 çıktı için değiştir
in_features = efficientnet.classifier[1].in_features
efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 4),
    nn.Sigmoid(),
)

efficientnet = efficientnet.to(DEVICE)

total     = sum(p.numel() for p in efficientnet.parameters())
trainable = sum(p.numel() for p in efficientnet.parameters() if p.requires_grad)
print(f"Toplam parametre   : {total:,}")
print(f"Egitilebilir       : {trainable:,}\n")

# ── Diferansiyel learning rate ────────────────────────────────
# Pretrained katmanlar çok düşük lr alır — ağırlıkları korur
# Yeni FC katmanı yüksek lr alır — sıfırdan öğrenir
backbone_params = [p for n, p in efficientnet.named_parameters()
                   if "classifier" not in n]
head_params     = list(efficientnet.classifier.parameters())

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
], weight_decay=1e-4)

criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=4, factor=0.5
)

# ── Eğitim döngüsü ────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0

print(f"{'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>10}")
print("-" * 46)

for epoch in range(1, EPOCHS + 1):

    efficientnet.train()
    train_losses = []
    for imgs, scores in train_loader:
        imgs, scores = imgs.to(DEVICE), scores.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(efficientnet(imgs), scores)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    efficientnet.eval()
    val_losses, val_maes = [], []
    with torch.no_grad():
        for imgs, scores in test_loader:
            imgs, scores = imgs.to(DEVICE), scores.to(DEVICE)
            preds = efficientnet(imgs)
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
        torch.save(efficientnet.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping - epoch {epoch}")
            break

# ── En iyi modeli yükle ve değerlendir ───────────────────────
print(f"\nEn iyi val loss: {best_val_loss:.5f}")
efficientnet.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
efficientnet.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, scores in test_loader:
        all_preds.append(efficientnet(imgs.to(DEVICE)).cpu())
        all_labels.append(scores)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# ── Wellbeing formülü ─────────────────────────────────────────
def compute_wellbeing(h, s, a, f):
    neg_weight = 1.0 - h * 0.6
    wb = (h * 0.55 + (1 - s * neg_weight) * 0.20
          + (1 - a * neg_weight) * 0.15 + (1 - f * neg_weight) * 0.10)
    return float(np.clip(wb, 0, 1) * 100)

# ── Metrikler ─────────────────────────────────────────────────
soft      = {"Happy":[0.85,0.05,0.05,0.05],"Sad":[0.05,0.82,0.05,0.08],
             "Angry":[0.05,0.10,0.75,0.10],"Fear":[0.05,0.15,0.05,0.75]}
test_df   = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
label_map = {"Happy":0,"Sad":1,"Angry":2,"Fear":3}

pred_cols = ["pred_happiness_score","pred_sadness_score",
             "pred_anger_score","pred_fear_score"]
col_to_label = {v: k for k, v in
                {"Happy":"pred_happiness_score","Sad":"pred_sadness_score",
                 "Angry":"pred_anger_score","Fear":"pred_fear_score"}.items()}

pred_df = test_df[["file_path","filename","label"]].copy()
for i, col in enumerate(SCORE_COLS):
    pred_df[f"pred_{col}"] = np.round(all_preds[:, i], 4)

pred_df["pred_psychological_wellbeing"] = [
    round(compute_wellbeing(*all_preds[i]), 1) for i in range(len(all_preds))
]
pred_df["pred_label"] = pred_df[pred_cols].idxmax(axis=1).map(col_to_label)

print(f"\n{'=== REGRESYON ==='}")
print(f"  {'Skor':<25} {'MAE':>8} {'R2':>8}")
print("  " + "-" * 43)
for i, name in enumerate(SCORE_COLS):
    y_t = np.array([soft[l][i] for l in test_df["label"]])
    mae = np.abs(all_preds[:, i] - y_t).mean()
    r2  = r2_score(y_t, all_preds[:, i])
    print(f"  {name:<25} {mae:>8.4f} {r2:>8.4f}")

print(f"\n{'=== SINIFLANDIRMA ==='}")
labels = ["Angry","Fear","Happy","Sad"]
f1s    = f1_score(test_df["label"], pred_df["pred_label"], labels=labels, average=None)
f1_mac = f1_score(test_df["label"], pred_df["pred_label"], average="macro")
acc    = (pred_df["pred_label"] == test_df["label"]).mean()

for label, f1 in zip(labels, f1s):
    print(f"  {label:<8} F1: {f1:.3f}")
print(f"  Macro F1 : {f1_mac:.3f}")
print(f"  Accuracy : %{acc*100:.1f}")

# ── Tahminleri kaydet ─────────────────────────────────────────
pred_df.to_csv(r"c:\ML_Project\test_predictions_v2.csv",
               index=False, encoding="utf-8-sig")
pred_df.to_excel(r"c:\ML_Project\test_predictions_v2.xlsx",
                 index=False, engine="openpyxl")

print(f"\nModel    : {MODEL_PATH}")
print(f"Tahminler: test_predictions_v2.csv / .xlsx")
