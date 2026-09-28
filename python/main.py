#!/usr/bin/env python3
"""Arduino App Lab entry point for OCR unoQ.

Runs the web preview (no monitor needed) on port 5001:
    http://<device-ip>:5001

Usage (from the app root directory):
    python3 python/main.py
Or via App Lab CLI:
    arduino-app-cli app start .
"""

import os
import sys

# Make the package importable when run from the app root.
# The package lives in <app-root>/src/ocr_unoq (same layout as on desktop).
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from ocr_unoq import Camera, TextReader, run_web_preview

RTSP_URL = "rtsp://10.0.0.124:554/mjpeg/1"


def main() -> None:
    camera = Camera(RTSP_URL, mirror=True)
    reader = TextReader()
    run_web_preview(camera, reader, port=5001)


if __name__ == "__main__":
    main()
