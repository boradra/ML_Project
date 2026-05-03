"""
Çocuk çizimlerini psikolojik açıdan skorlayan ve train/test olarak ayıran pipeline.

Adımlar:
  1. Her görsel için renk/uzamsal özellikler çıkarılır (PIL + NumPy)
  2. Klasör etiketi + görsel özellikler → psikolojik skorlar hesaplanır
  3. Stratified 80/20 train-test split uygulanır
  4. Görseller dataset/train/ ve dataset/test/ klasörlerine kopyalanır
  5. scored_dataset.csv, train_labels.csv, test_labels.csv oluşturulur
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATASET_PATH = r"C:\Users\borad\.cache\kagglehub\datasets\vishmiperera\children-drawings\versions\1"
OUTPUT_DIR   = r"c:\ML_Project\dataset"
CSV_ALL      = r"c:\ML_Project\scored_dataset.csv"
CSV_TRAIN    = r"c:\ML_Project\train_labels.csv"
CSV_TEST     = r"c:\ML_Project\test_labels.csv"
XLSX_ALL     = r"c:\ML_Project\scored_dataset.xlsx"
XLSX_TRAIN   = r"c:\ML_Project\train_labels.xlsx"
XLSX_TEST    = r"c:\ML_Project\test_labels.xlsx"

LABELS = ("Angry", "Fear", "Happy", "Sad")

# ──────────────────────────────────────────────────────────────
# 1. Görsel Özellik Çıkarımı
# ──────────────────────────────────────────────────────────────

def extract_visual_features(img_path: str) -> dict:
    """
    PIL ile görseli yükler; HSV renk uzayında istatistikler hesaplar.
    Döndürülen değerler 0-1 arasında normalize edilmiştir.
    """
    try:
        img = Image.open(img_path).convert("RGB").resize((128, 128))
    except Exception:
        return _default_features()

    arr = np.array(img, dtype=np.float32) / 255.0   # (128,128,3) RGB 0-1

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # HSV benzeri hesaplamalar (PIL'siz, NumPy ile)
    cmax = np.max(arr, axis=2)
    cmin = np.min(arr, axis=2)
    delta = cmax - cmin

    # Parlaklık (Value)
    brightness = float(np.mean(cmax))

    # Doygunluk (Saturation)
    sat = np.where(cmax > 0, delta / (cmax + 1e-6), 0)
    saturation = float(np.mean(sat))

    # Sıcak renk oranı: kırmızı/turuncu/sarı baskın pikseller
    warm_mask = (r > 0.5) & (r > g * 1.2) & (r > b * 1.2)
    warm_ratio = float(np.mean(warm_mask))

    # Soğuk renk oranı: mavi/mor baskın
    cool_mask = (b > 0.4) & (b > r * 1.1)
    cool_ratio = float(np.mean(cool_mask))

    # Kırmızı-agresyon oranı: çok yoğun kırmızı
    red_mask = (r > 0.6) & (r > g + 0.2) & (r > b + 0.2)
    red_ratio = float(np.mean(red_mask))

    # Karanlık piksel oranı
    dark_mask = cmax < 0.25
    dark_ratio = float(np.mean(dark_mask))

    # Boş (beyaz) alan oranı → ters çevrilince alan kullanımı
    white_mask = (r > 0.85) & (g > 0.85) & (b > 0.85)
    space_usage = 1.0 - float(np.mean(white_mask))

    # Renk çeşitliliği: benzersiz renk kümeleri için histogram entropisi
    hist_r = np.histogram(r.flatten(), bins=16, range=(0,1))[0]
    hist_g = np.histogram(g.flatten(), bins=16, range=(0,1))[0]
    hist_b = np.histogram(b.flatten(), bins=16, range=(0,1))[0]
    hist = (hist_r + hist_g + hist_b).astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
    color_diversity = min(entropy / 4.0, 1.0)   # 4.0 = maksimum entropi normalize

    return {
        "brightness":    round(brightness, 4),
        "saturation":    round(saturation, 4),
        "warm_ratio":    round(warm_ratio, 4),
        "cool_ratio":    round(cool_ratio, 4),
        "red_ratio":     round(red_ratio, 4),
        "dark_ratio":    round(dark_ratio, 4),
        "space_usage":   round(space_usage, 4),
        "color_diversity": round(color_diversity, 4),
    }


def _default_features() -> dict:
    return {k: 0.5 for k in
            ("brightness","saturation","warm_ratio","cool_ratio",
             "red_ratio","dark_ratio","space_usage","color_diversity")}


# ──────────────────────────────────────────────────────────────
# 2. Psikolojik Skorlama
# ──────────────────────────────────────────────────────────────

def compute_psych_scores(vf: dict) -> dict:
    """
    Görsel özelliklerden psikolojik proxy skorları üretir (etiket bağımsız).
    Etkileşim terimleri kullanılır: tek özellik değil, birleşik sinyal önemlidir.
    Psikolog verileri hazır olduğunda bu skorlar gerçek annotasyonla değiştirilecek.
    """
    b   = vf["brightness"]
    sat = vf["saturation"]
    w   = vf["warm_ratio"]
    c   = vf["cool_ratio"]
    r   = vf["red_ratio"]
    dk  = vf["dark_ratio"]
    sp  = vf["space_usage"]
    cd  = vf["color_diversity"]

    def clip(x): return round(float(np.clip(x, 0.0, 1.0)), 4)

    # ── Etkileşim terimleri ───────────────────────────────────
    warm_bright    = w * b              # sıcak VE parlak: pozitif enerji
    cool_dark      = c * dk             # serin VE karanlık: içe kapanma
    dark_red       = r * (1 - b)        # karanlık kırmızı: öfke sinyali (aydınlık kırmızı değil)
    darkness_depth = dk * (1 - b)       # salt karanlık piksel değil, genel atmosfer karanlığı
    tense_saturation = sat * dk         # yoğun doygun + karanlık: duygusal gerilim

    # ── Happiness ────────────────────────────────────────────
    # Sıcak+parlak birlikteliği, renk zenginliği ve alan dolulumu birlikte yüksekse mutluluk
    happiness = clip(
        warm_bright  * 0.38 +
        cd           * 0.22 +
        sp           * 0.18 +
        b            * 0.14 +
        (1 - dk)     * 0.08
    )

    # ── Sadness ──────────────────────────────────────────────
    # Serin+karanlık birlikteliği, mat renkler ve küçülen alan kullanımı
    sadness = clip(
        cool_dark    * 0.30 +
        (1 - sat)    * 0.22 +
        dk           * 0.20 +
        (1 - sp)     * 0.16 +
        (1 - b)      * 0.12
    )

    # ── Anger ────────────────────────────────────────────────
    # Karanlık kırmızı yoğunluğu belirleyici; doygun+karanlık gerilim ve monoton renk bunu destekler
    anger = clip(
        dark_red         * 0.38 +
        r                * 0.24 +
        tense_saturation * 0.20 +
        (1 - cd)         * 0.18
    )

    # ── Fear ─────────────────────────────────────────────────
    # Atmosferik karanlık derinliği + serin soğukluk + kısıtlı alan + renk yoksulluğu
    fear = clip(
        darkness_depth * 0.35 +
        c * (1 - b)    * 0.25 +
        (1 - sp)       * 0.22 +
        (1 - cd)       * 0.18
    )

    # ── Psikolojik iyilik skoru (0-100) ──────────────────────
    # Pozitif valans + görsel zenginlik eksi duygusal gerilim (geometrik ortalama)
    positive_valence  = (happiness * 0.50 + (1 - sadness) * 0.20
                         + (1 - fear) * 0.15 + (1 - anger) * 0.15)
    visual_richness   = cd * 0.40 + sp * 0.35 + b * 0.25
    emotional_tension = (anger * fear) ** 0.5   # her ikisi yüksekse gerçek gerilim
    raw_wb = (positive_valence * 0.60 + visual_richness * 0.30 - emotional_tension * 0.10) * 100
    psychological_wellbeing = int(np.clip(round(raw_wb), 0, 100))

    return {
        "happiness_score":         happiness,
        "sadness_score":           sadness,
        "anger_score":             anger,
        "fear_score":              fear,
        "brightness":              clip(b),
        "saturation":              clip(sat),
        "warm_ratio":              clip(w),
        "cool_ratio":              clip(c),
        "red_ratio":               clip(r),
        "dark_ratio":              clip(dk),
        "space_usage":             clip(sp),
        "color_diversity":         clip(cd),
        "psychological_wellbeing": psychological_wellbeing,
    }


# ──────────────────────────────────────────────────────────────
# 3. Dataset Toplama
# ──────────────────────────────────────────────────────────────

def collect_images() -> list[dict]:
    records = []
    for root, _, files in os.walk(DATASET_PATH):
        parent = os.path.basename(root)
        if parent not in LABELS:
            continue
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            full_path = os.path.join(root, fname)
            records.append({"file_path": full_path, "filename": fname, "label": parent})
    return records


# ──────────────────────────────────────────────────────────────
# 4. Ana Pipeline
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ÇOCUK ÇİZİMLERİ — PSİKOLOJİK SKORLAMA & TRAIN/TEST SPLIT")
    print("=" * 60)

    # ── Görsel topla ─────────────────────────────────────────
    records = collect_images()
    print(f"\nToplam görsel: {len(records)}")

    df = pd.DataFrame(records)   # file_path, filename, label
    print(f"\nToplanan görsel: {df.shape[0]}")

    # ── Train / Test split (stratified, 80/20) ───────────────
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"
    df_all = pd.concat([train_df, test_df]).sort_index()

    print(f"\nTrain: {len(train_df)} görsel")
    print(f"Test : {len(test_df)} görsel")
    print("\nLabel dağılımı:")
    print(df_all.groupby(["split", "label"]).size().to_string())

    # ── CSV kaydet (sadece gerekli sütunlar) ─────────────────
    COLS = ["file_path", "filename", "label"]

    df_all[COLS + ["split"]].to_csv(CSV_ALL,   index=False, encoding="utf-8-sig")
    train_df[COLS].to_csv(CSV_TRAIN,           index=False, encoding="utf-8-sig")
    test_df[COLS].to_csv(CSV_TEST,             index=False, encoding="utf-8-sig")
    df_all[COLS + ["split"]].to_excel(XLSX_ALL,   index=False, engine="openpyxl")
    train_df[COLS].to_excel(XLSX_TRAIN,           index=False, engine="openpyxl")
    test_df[COLS].to_excel(XLSX_TEST,             index=False, engine="openpyxl")
    print(f"\nKaydedildi: {CSV_ALL}, {CSV_TRAIN}, {CSV_TEST}")

    # ── Görselleri klasörlere kopyala ─────────────────────────
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        for label in LABELS:
            dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)

        subset = split_df[["file_path", "filename", "label"]]
        for _, row in subset.iterrows():
            dest = os.path.join(OUTPUT_DIR, split_name, row["label"], row["filename"])
            # Aynı isimde dosya varsa üzerine yaz
            shutil.copy2(row["file_path"], dest)

    print(f"\nGorseller kopyalandi: {OUTPUT_DIR}")
    print("  dataset/")
    for split_name in ("train", "test"):
        for label in LABELS:
            d = os.path.join(OUTPUT_DIR, split_name, label)
            n = len(os.listdir(d))
            print(f"    {split_name}/{label}/  ({n} görsel)")

    print("\nTamamlandı!")


if __name__ == "__main__":
    main()
