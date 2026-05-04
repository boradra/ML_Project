"""
Claude Vision API ile çocuk çizimlerini bağımsız 0-1 skor olarak puanlar.

Her duygu bağımsız değerlendirilir — toplamı 1 olması şart değil.
Model: claude-haiku-4-5 (696 görsel için maliyet etkin)
Çıktı: train_scores_new.xlsx
"""

import anthropic
import base64
import json
import os
import time
import pandas as pd
from pathlib import Path

# ── Ayarlar ───────────────────────────────────────────────────
TRAIN_CSV   = r"c:\ML_Project\train_labels.csv"
OUTPUT_XLSX = r"c:\ML_Project\train_scores_new.xlsx"
PROGRESS_CSV = r"c:\ML_Project\score_progress.csv"
MODEL       = "claude-haiku-4-5"
DELAY_SEC   = 0.3   # rate limit için bekleme

SYSTEM_PROMPT = """Sen çocuk çizimlerini psikolojik açıdan değerlendiren bir uzmansın.
Senden her çizim için 4 duyguyu BAĞIMSIZ olarak 0-1 arasında puanlamanı istiyorum.

Önemli kurallar:
- Her duygu diğerinden bağımsız olarak değerlendirilir
- Puanlar toplanarak 1 olmak ZORUNDA değil
- Bir çizim hem mutlu hem korkmuş olabilir
- 0: Bu duygu hiç yok  |  1: Bu duygu çok güçlü

Puanlama rehberi:
- happiness_score: Neşe, gülümseme, renkli/parlak renkler, pozitif figürler
- sadness_score: Üzüntü, gözyaşı, koyu renkler, yalnız figürler, düşük enerji
- anger_score: Öfke, kaotik çizgiler, kırmızı/siyah ağırlıklı, agresif şekiller
- fear_score: Korku, kaçma figürleri, tehdit unsurları, karanlık atmosfer"""

def encode_image(path: str) -> tuple[str, str]:
    """Görseli base64'e çevirir, media type döner."""
    suffix = Path(path).suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type

def score_image(client: anthropic.Anthropic, image_path: str) -> dict:
    """Tek bir görseli Claude ile puanlar."""
    img_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Bu çocuk çizimini değerlendir ve aşağıdaki JSON formatında yanıt ver:\n"
                        '{"happiness_score": 0.0, "sadness_score": 0.0, '
                        '"anger_score": 0.0, "fear_score": 0.0}\n'
                        "Sadece JSON döndür, başka bir şey yazma."
                    )
                }
            ]
        }],
    )

    text = response.content[0].text.strip()
    # JSON bloğunu çıkar
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def main():
    client = anthropic.Anthropic()

    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    total = len(df)
    print(f"Toplam görsel: {total}")

    # Daha önce tamamlananları yükle
    if os.path.exists(PROGRESS_CSV):
        done_df = pd.read_csv(PROGRESS_CSV, encoding="utf-8-sig")
        done_files = set(done_df["filename"].tolist())
        print(f"Önceden tamamlanan: {len(done_files)}")
    else:
        done_df = pd.DataFrame()
        done_files = set()

    results = done_df.to_dict("records") if not done_df.empty else []
    errors  = []

    remaining = df[~df["filename"].isin(done_files)]
    print(f"Kalan: {len(remaining)}\n")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        filename   = row["filename"]
        file_path  = row["file_path"]
        label      = row["label"]

        print(f"[{i}/{len(remaining)}] {filename} ({label})...", end=" ", flush=True)

        try:
            scores = score_image(client, file_path)
            results.append({
                "filename": filename,
                "label": label,
                "happiness_score": round(scores["happiness_score"], 3),
                "sadness_score":   round(scores["sadness_score"],   3),
                "anger_score":     round(scores["anger_score"],     3),
                "fear_score":      round(scores["fear_score"],      3),
            })
            print(
                f"H={scores['happiness_score']:.2f} "
                f"S={scores['sadness_score']:.2f} "
                f"A={scores['anger_score']:.2f} "
                f"F={scores['fear_score']:.2f}"
            )

            # Her 50 görselde bir ilerlemeyi kaydet
            if i % 50 == 0:
                pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")
                print(f"  >> İlerleme kaydedildi ({len(results)} tamamlandı)")

        except Exception as e:
            print(f"HATA: {e}")
            errors.append({"filename": filename, "error": str(e)})

        time.sleep(DELAY_SEC)

    # Sonuçları kaydet
    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    result_df.to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"Tamamlandı: {len(results)} / {total}")
    print(f"Hata: {len(errors)}")
    print(f"Sonuçlar: {OUTPUT_XLSX}")

    # İstatistikler
    cols = ["happiness_score", "sadness_score", "anger_score", "fear_score"]
    print("\n=== SKOR İSTATİSTİKLERİ ===")
    print(result_df[cols].describe().round(3))

    toplam = result_df[cols].sum(axis=1)
    print(f"\nSkor toplamı: ort={toplam.mean():.3f}  std={toplam.std():.3f}")
    print("(Bağımsız skorlama → toplam 1 olmak zorunda değil)")

if __name__ == "__main__":
    main()
