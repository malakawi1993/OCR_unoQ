"""Background OCR engine — camera + OCR run on a background thread.

For kids:

    from ocr_unoq import Camera, TextReader, LiveOCR

    camera = Camera("rtsp://10.0.0.124:554/mjpeg/1")   # or Camera(0) for USB
    reader = TextReader()

    with LiveOCR(camera, reader) as live:
        while True:
            frame = live.frame          # latest camera frame (None if dead)
            results = live.results      # text from the last OCR pass
            ...your drawing / printing code...

No threading code needed — LiveOCR does it all for you.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Self


class LiveOCR:
    """Runs the camera + OCR on a background thread.

    Your code just reads ``live.frame`` and ``live.results`` as often as
    it likes — the preview never freezes while inference is running.

    Args:
        camera: A Camera (or anything with ``.grab()`` / ``.release()``).
        reader: A TextReader (or anything callable on a frame).
        ocr_interval: Seconds between OCR passes (default 1.5).
    """

    def __init__(self, camera: Any, reader: Any, ocr_interval: float = 1.5) -> None:
        self._camera = camera
        self._reader = reader
        self.ocr_interval = ocr_interval

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._grab_thread: threading.Thread | None = None
        self._ocr_thread: threading.Thread | None = None

        self._frame: Any = None
        self._results: list = []
        self._elapsed_ms: float = 0.0
        self._last_run: float = 0.0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> Self:
        """Start the background threads (no-op if already running)."""
        if self._grab_thread is not None:
            return self
        self._stop_event.clear()
        # Two threads so OCR never slows down the video feed:
        #   grabber — pulls frames as fast as the camera sends them
        #   ocr     — reads text from the latest frame on a timer
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
        self._grab_thread.start()
        self._ocr_thread.start()
        return self

    def stop(self) -> None:
        """Stop the background threads and wait for them to finish."""
        self._stop_event.set()
        for thread in (self._grab_thread, self._ocr_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        self._grab_thread = None
        self._ocr_thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- thread-safe accessors ----------------------------------------------

    @property
    def frame(self) -> Any:
        """The latest camera frame (None if the stream is dead)."""
        with self._lock:
            return self._frame

    @property
    def results(self) -> list:
        """Text found in the most recent OCR pass."""
        with self._lock:
            return list(self._results)

    @property
    def elapsed_ms(self) -> float:
        """How long the last OCR pass took, in milliseconds."""
        with self._lock:
            return self._elapsed_ms

    @property
    def last_run(self) -> float:
        """monotonic() time of the last OCR pass (0 if never)."""
        with self._lock:
            return self._last_run

    # -- internals ------------------------------------------------------------

    def _grab_loop(self) -> None:
        """Pull frames as fast as the camera sends them (keeps the feed live)."""
        while not self._stop_event.is_set():
            frame = self._camera.grab()
            if frame is None:
                time.sleep(0.25)
                continue
            with self._lock:
                self._frame = frame

    def _ocr_loop(self) -> None:
        """Read text from the latest frame on a timer (never blocks the feed)."""
        last_ocr = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_ocr < self.ocr_interval:
                time.sleep(0.1)
                continue

            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.25)
                continue

            last_ocr = now
            started = time.perf_counter()
            try:
                results = self._reader(frame)
            except Exception as exc:  # never kill the preview  # noqa: BLE001
                print(f"OCR failed: {exc}")
                results = []
            elapsed_ms = (time.perf_counter() - started) * 1000

            with self._lock:
                self._results = results
                self._elapsed_ms = elapsed_ms
                self._last_run = now
