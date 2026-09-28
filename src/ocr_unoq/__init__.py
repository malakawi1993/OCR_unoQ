"""ocr_unoq — read text out of camera frames and images.

Made for the Arduino UNO Q, but works on any computer.

The three things you need to know:

    from ocr_unoq import Camera, TextReader, read_text

    # 1) Read text from a photo file:
    results = read_text("photo.jpg")
    print(results[0].text if results else "no text")

    # 2) Read text from a live camera (runs on a background thread —
    #    your loop never freezes while OCR is thinking):
    from ocr_unoq import LiveOCR

    camera = Camera("rtsp://10.0.0.124:554/mjpeg/1")   # or Camera(0) for USB
    reader = TextReader()
    with LiveOCR(camera, reader) as live:
        while True:
            frame = live.frame          # latest camera frame
            results = live.results      # text from the last OCR pass

    # 3) Show a live preview in a browser (for the UNO Q, no monitor needed):
    from ocr_unoq import run_web_preview
    run_web_preview(camera, reader, port=5001)
"""

from .camera import Camera
from .live import LiveOCR
from .ocr import (
    TextReader,
    TextResult,
    draw_results,
    get_reader,
    read_text,
    status_line,
)
from .webview import run_web_preview

__all__ = [
    "Camera",
    "LiveOCR",
    "TextReader",
    "TextResult",
    "draw_results",
    "get_reader",
    "read_text",
    "run_web_preview",
    "status_line",
]
