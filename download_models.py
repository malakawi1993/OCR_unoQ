"""One-time download of the OCR models from HuggingFace.

Run this once before main.py:
    python download_models.py

Downloads two model sets (~107 MB total):

  v5 — accurate but heavy (best for desktops):
    detection/v5/det.onnx          (84 MB) finds text boxes
    languages/english/rec.onnx     (7.5 MB) reads English text
    languages/english/dict.txt            the character set for decoding

  v4 — small and fast (best for the Arduino UNO Q):
    detection/v4/det.onnx          (~4.5 MB) finds text boxes
    languages/english/v4/rec.onnx  (~10 MB) reads English text
    languages/english/v4/dict.txt         the character set for decoding

The v4 files are mirrors of PaddlePaddle's official ONNX exports
(cycloneboy/*). The det model is language-agnostic — "ch_" is just
PaddleOCR's naming for its default detector.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# main.py expects the models inside the package, not in the repo root.
MODELS_DIR = Path(__file__).resolve().parent / "src" / "ocr_unoq" / "models"

# --- v5 (monkt/paddleocr-onnx) ---------------------------------------------

paths = snapshot_download(
    repo_id="monkt/paddleocr-onnx",
    allow_patterns=[
        "detection/v5/det.onnx",
        "languages/english/rec.onnx",
        "languages/english/dict.txt",
    ],
    local_dir=str(MODELS_DIR),
)
print("v5 models:", paths)

# --- v4 (cycloneboy mirrors of the official PaddlePaddle ONNX exports) ------

V4_FILES = [
    # (repo_id, filename in repo, destination under MODELS_DIR)
    ("cycloneboy/ch_PP-OCRv4_det_infer", "model.onnx", "detection/v4/det.onnx"),
    ("cycloneboy/en_PP-OCRv4_rec_infer", "model.onnx", "languages/english/v4/rec.onnx"),
    ("cycloneboy/en_PP-OCRv4_rec_infer", "en_dict.txt", "languages/english/v4/dict.txt"),
]

for repo_id, filename, dest in V4_FILES:
    target = MODELS_DIR / dest
    if target.exists():
        print(f"skip (already there): {dest}")
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, target)
    print(f"downloaded: {dest} ({target.stat().st_size / 1e6:.1f} MB)")

print("All models ready in:", MODELS_DIR)
