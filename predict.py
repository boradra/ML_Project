"""
Yeni bir çocuk çizimi için psikolojik skor tahmini.

Kullanım:
  python predict.py resim.jpg
  python predict.py C:/yol/resim.png
  python predict.py C:/klasor/*.jpg        (toplu tahmin)
"""

import sys
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models
import torch.nn as nn
from dataset import VAL_TRANSFORM, SCORE_COLS

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"c:\ML_Project\model_resnet.pt"

# ── Model yükle ───────────────────────────────────────────────
def load_model():
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 4),
        nn.Sigmoid(),
    )
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    resnet.to(DEVICE).eval()
    return resnet

# ── Tek resim tahmini ─────────────────────────────────────────
def predict_image(model, img_path: str) -> dict:
    img    = Image.open(img_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor).cpu().numpy()[0]

    scores = {col: round(float(out[i]), 4) for i, col in enumerate(SCORE_COLS)}

    # Wellbeing: dinamik ağırlıklı formül
    h, s, a, f = scores["happiness_score"], scores["sadness_score"], \
                 scores["anger_score"],     scores["fear_score"]
    neg_weight = 1.0 - h * 0.6
    wb = (h * 0.55 + (1 - s * neg_weight) * 0.20
          + (1 - a * neg_weight) * 0.15 + (1 - f * neg_weight) * 0.10)
    scores["psychological_wellbeing"] = round(float(np.clip(wb, 0, 1) * 100), 1)
    scores["dominant_emotion"] = max(SCORE_COLS, key=lambda c: scores[c]).replace("_score", "").capitalize()
    return scores

# ── Sonuç yazdır ──────────────────────────────────────────────
def print_result(img_path: str, scores: dict):
    print(f"\nResim : {img_path}")
    print(f"  Dominant duygu     : {scores['dominant_emotion']}")
    print(f"  Happiness          : {scores['happiness_score']:.4f}")
    print(f"  Sadness            : {scores['sadness_score']:.4f}")
    print(f"  Anger              : {scores['anger_score']:.4f}")
    print(f"  Fear               : {scores['fear_score']:.4f}")
    print(f"  Psychological Well : {scores['psychological_wellbeing']:.1f} / 100")

# ── Ana akış ─────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python predict.py <resim_yolu>")
        print("Ornek  : python predict.py C:\\resim.jpg")
        sys.exit(1)

    # Glob ile çoklu dosya desteği (*.jpg gibi)
    pattern = sys.argv[1]
    paths   = glob.glob(pattern) or [pattern]

    if not paths:
        print(f"Dosya bulunamadi: {pattern}")
        sys.exit(1)

    model = load_model()
    print(f"Model yuklendi ({DEVICE})")

    results = []
    for path in paths:
        try:
            scores = predict_image(model, path)
            print_result(path, scores)
            results.append({"file_path": path, **scores})
        except Exception as e:
            print(f"HATA [{path}]: {e}")

    # Birden fazla resim varsa CSV'ye kaydet
    if len(results) > 1:
        out_csv = r"c:\ML_Project\predictions.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n{len(results)} resim icin tahminler kaydedildi: {out_csv}")
