"""
download_models.py
==================
Run this once before launching the app, OR let app.py handle it automatically.

Usage:
    pip install gdown
    python download_models.py
"""

import os
import gdown

# ── Google Drive file IDs — change these if you re-upload the models ──────────
MODELS = {
    "models/resnet50_best_phase2.keras" : "136sgvnBXCzP2C9jsDkj3r-BZhGrCEhCs",
    "models/mobilenetv2_maize_model.h5" : "1Fnuy-4pmwTsFXo_anJEDiLrsitYdR4Im",
    "models/best_phase2.keras"          : "1tLlZLChvxUOtGkHLVHW9rWWNf0zGhWe-",
}

os.makedirs("models", exist_ok=True)

for output_path, file_id in MODELS.items():
    if os.path.exists(output_path):
        print(f"✅ Already exists: {output_path} — skipping.")
        continue

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️  Downloading → {output_path}")
    gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1_000_000
        print(f"✅ {output_path}  ({size_mb:.1f} MB)\n")
    else:
        print(f"❌ Download failed for {output_path}. Check the file ID or Drive sharing settings.\n")

print("Done.")