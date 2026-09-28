# OCR_unoQ — read text out of camera frames, made for the Arduino UNO Q

A small Python **library** that reads an RTSP camera stream and finds any
text in it, using lightweight ONNX models that run on CPU only. Built to fit
in ~2GB of RAM (Arduino UNO Q / Raspberry Pi / Orange Pi).

## Quick start (the whole API)

```python
from ocr_unoq import Camera, TextReader, read_text

# Read text from a photo file:
results = read_text("photo.jpg")
print(results[0].text if results else "no text")

# Read text from a live camera (background thread — your loop never freezes):
from ocr_unoq import LiveOCR

camera = Camera("rtsp://10.0.0.124:554/mjpeg/1")   # or Camera(0) for USB cam
reader = TextReader()
with LiveOCR(camera, reader) as live:
    while True:
        frame = live.frame          # latest camera frame (None if stream dead)
        results = live.results      # text from the last OCR pass
        for r in results:
            print(r.text, r.confidence, r.center)  # center = (x, y) to point at
```

That's it — no configuration needed. Every parameter has a sensible default.

## The idea (how it works)

```text
camera ──RTSP──> OpenCV frame ──downscale──> RapidOCR ──> text lines
                                              │
                              1. DETECTION: where is the text? (boxes)
                              2. RECOGNITION: what does each box say?
```

- **`Camera`** pulls frames from an RTSP stream (or USB camera) and
  automatically reconnects if the stream drops.
- **`LiveOCR`** runs the camera + OCR on a background thread for you —
  your code just reads `live.frame` and `live.results`, no threading needed.
- **`TextReader`** runs two small neural networks on CPU via `onnxruntime`:
  - *detection* finds bounding boxes around text,
  - *recognition* reads each cropped box into characters.
- The models come from [monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)
  (PaddleOCR PP-OCRv5 converted to ONNX). RapidOCR is just the engine that runs them.

### Why it's light enough for a 2GB SBC

| Trick                     | Where                    | Effect                                                 |
| ------------------------- | ------------------------ | ------------------------------------------------------ |
| ONNX Runtime, CPU only    | `pyproject.toml`         | no GPU, no PaddlePaddle/PyTorch (~300–500MB RAM total) |
| Small v4 models (default) | `download_models.py`     | ~12 MB vs 92 MB — much faster on A53 cores             |
| Downscale frame to 640px  | `TextReader.max_width`   | OCR cost scales with pixels — biggest speedup          |
| OCR on a timer (1.5s)     | `LiveOCR`                | text rarely changes; saves most of the CPU             |
| Grab throttled to 12 fps  | `LiveOCR(max_fps=...)`   | no decoding frames nobody will ever see                |
| OCR limited to 2 threads  | `TextReader(intra_op_threads=...)` | leaves cores for the camera feed + web stream |
| 640px preview JPEGs       | `run_web_preview(preview_width=...)` | encoding small JPEGs is far cheaper on A53 cores |
| Auto-reconnecting camera  | `Camera.grab()`          | RTSP streams drop; it reopens them for you             |

## Setup (desktop)

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra models             # install dependencies + huggingface-hub
uv run python download_models.py   # one-time, ~92 MB into src/ocr_unoq/models
uv run python src/ocr_unoq/main.py # live preview window (press q to quit)
```

Set `RTSP_URL` at the top of `src/ocr_unoq/main.py` to your camera's stream.
The status line at the bottom of the window shows when the last OCR pass ran
and what it found (including "no text found"), so you can tell OCR is working
even on a quiet scene.

## Running on the Arduino UNO Q (Arduino App Lab)

The UNO Q has no monitor, so instead of a preview window the app serves a
**web preview** — open `http://<board-ip>:5001` in any browser.

```bash
# 1) Copy the project to the board:
scp -r . arduino@<device-ip>:/home/arduino/ArduinoApps/ocr-unoq

# 2) SSH in and set up:
ssh arduino@<device-ip>
cd /home/arduino/ArduinoApps/ocr-unoq
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r python/requirements.txt   # headless OpenCV + Flask
python download_models.py                               # one-time, ~92 MB

# 3) Run it:
python3 python/main.py          # or: arduino-app-cli app start .
```

Then point your browser at `http://<device-ip>:5001` — you'll see the live
camera feed with text boxes and a status line, updating in real time.

> The UNO Q uses `opencv-python-headless` (no GUI libraries) via
> `python/requirements.txt`; desktops use regular `opencv-python` for the
> preview window. Everything else is identical.

## Tuning for your hardware

Everything is optional — pass what you need when creating a `TextReader`:

```python
reader = TextReader(
    min_confidence=0.40,   # drop results below this score (default 0.40)
    box_thresh=0.5,        # detection sensitivity; lower = finds faint text
    max_width=640,         # frames are downscaled to this width before OCR
    intra_op_threads=2,    # CPU threads for OCR (-1 = all cores, desktops)
    inter_op_threads=1,
)
camera = Camera(url, mirror=True)   # flip if your camera is mirrored
```

- **Feed stalling / CPU at 100% (UNO Q)?** the defaults already save CPU:
  OCR is limited to 2 threads, `LiveOCR` decodes at most 12 fps, and the web
  preview encodes 640px JPEGs. If it's still hot, lower them further:
  `run_web_preview(..., max_fps=8, preview_width=480)` and
  `TextReader(intra_op_threads=1)`. On a desktop, set
  `intra_op_threads=-1` to use all cores for faster OCR.
- **Too slow?** run OCR less often (raise the interval in your loop), or lower `max_width`.
- **Missing small text?** raise `max_width` (e.g. 960) — costs CPU.
- **Garbage results?** lower `min_confidence` to 0.3, or check the camera angle/lighting.
- **Debugging?** save a frame and OCR it: `read_text("frame.png")` — if that
  works but the live stream doesn't, the problem is blur/lighting, not the code.

## Even lighter option (if 92MB models are too slow on your SBC)

RapidOCR ships tiny "mobile" PaddleOCR models built in — zero download:

```python
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()   # uses bundled PP-OCRv4 mobile det+rec (~10 MB total)
```

Accuracy is a bit lower than the monkt v5 models, but detection is ~30x smaller
(2.3MB vs 84MB) and much faster on weak CPUs. Try this first if your SBC stuggles.

## Files

```
src/ocr_unoq/
├── __init__.py     # public API: Camera, LiveOCR, TextReader, read_text, run_web_preview
├── ocr.py          # TextReader + TextResult (the OCR engine)
├── camera.py       # Camera with auto-reconnect
├── live.py         # LiveOCR — background-thread camera + OCR engine
├── webview.py      # browser preview for the UNO Q (Flask, port 5001)
├── main.py         # desktop demo: live preview window
└── models/         # downloaded ONNX models (~92 MB)
app.yaml            # Arduino App Lab config (port 5001)
python/main.py      # App Lab entry point (web preview on the board)
python/requirements.txt  # UNO Q deps (headless OpenCV + Flask)
download_models.py  # one-time model download from HuggingFace
```
