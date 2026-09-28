"""A tiny web page that shows the camera + OCR results in a browser.

Use this on the Arduino UNO Q (or anywhere without a monitor):

    from ocr_unoq import Camera, TextReader, run_web_preview

    camera = Camera("rtsp://10.0.0.124:554/mjpeg/1", mirror=True)
    reader = TextReader()
    run_web_preview(camera, reader, port=5001)   # then open http://<board-ip>:5001

The page auto-refreshes with the latest frame, text boxes, and a status line.
"""

from __future__ import annotations

import time
from typing import Any

from .live import LiveOCR

DEFAULT_PORT = 5001  # App Lab convention (port 7000 is reserved)


def run_web_preview(
    camera: Any,
    reader: Any,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    ocr_interval: float = 1.5,
) -> None:
    """Start the web preview and block until Ctrl+C.

    Args:
        camera: A Camera instance (or anything with .grab() and .release()).
        reader: A TextReader instance (or anything callable on a frame).
        host: Address to listen on.
        port: Port for the web page.
        ocr_interval: Seconds between OCR passes.
    """
    from flask import Flask, Response

    app = Flask(__name__)

    live = LiveOCR(camera, reader, ocr_interval=ocr_interval).start()

    def generate():
        import cv2

        while True:
            frame = live.frame
            results = live.results
            elapsed_ms = live.elapsed_ms
            last_run = live.last_run

            if frame is None:
                time.sleep(0.5)
                continue

            from .ocr import draw_results, status_line

            display = draw_results(frame, results)

            age = (
                f"{time.monotonic() - last_run:.1f}s ago" if last_run else "never"
            )
            cv2.putText(
                display,
                f"{status_line(results, elapsed_ms)} | last pass {age}",
                (8, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            ok, jpeg = cv2.imencode(".jpg", display)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

            time.sleep(0.1)  # ~10 fps preview

    @app.route("/")
    def index():
        return (
            "<html><head><title>OCR Preview</title></head>"
            "<body style='margin:0;background:#222'>"
            "<img src='/stream' style='width:100%'>"
            "</body></html>"
        )

    @app.route("/stream")
    def stream():
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Web preview: http://{host}:{port}  (Ctrl+C to stop)")
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        live.stop()
        camera.release()
