"""Demo: live OCR preview window (for desktops with a monitor).

Run with:  uv run python src/ocr_unoq/main.py

On the Arduino UNO Q (no monitor) use the web preview instead — see
python/main.py in this repo, or call ``run_web_preview`` from your own code.
"""

import time

from ocr_unoq import Camera, LiveOCR, TextReader, draw_results, status_line

RTSP_URL = "rtsp://10.0.0.124:554/mjpeg/1"

# Tuning knobs (all optional — the defaults work fine):
MIRROR_CORRECTION = True
OCR_INTERVAL_SECONDS = 2.0
MIN_CONFIDENCE = 0.40
OCR_MAX_WIDTH = 640
DET_BOX_THRESH = 0.5


def main() -> None:
    import cv2

    camera = Camera(RTSP_URL, mirror=MIRROR_CORRECTION)
    reader = TextReader(
        min_confidence=MIN_CONFIDENCE,
        box_thresh=DET_BOX_THRESH,
        max_width=OCR_MAX_WIDTH,
    )

    print("Waiting for frames... (press q in the window to quit)")

    # LiveOCR runs camera + OCR on a background thread, so this loop
    # stays smooth even while inference is running.
    live = LiveOCR(camera, reader, ocr_interval=OCR_INTERVAL_SECONDS)
    try:
        live.start()
        while True:
            frame = live.frame
            if frame is None:
                cv2.waitKey(1)
                continue

            results = live.results
            display = draw_results(frame, results)

            age = (
                f"{time.monotonic() - live.last_run:.1f}s ago"
                if live.last_run
                else "never"
            )
            cv2.putText(
                display,
                f"{status_line(results, live.elapsed_ms)} | last pass {age}",
                (8, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            cv2.imshow("OCR Preview", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Shutdown order matters: stop the background threads FIRST, then
        # release the camera (releasing it mid-read crashes OpenCV).
        live.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
