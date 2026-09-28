"""A simple camera that grabs frames and reconnects if the stream drops.

For kids:

    from ocr_unoq import Camera

    camera = Camera("rtsp://10.0.0.124:554/mjpeg/1")   # or Camera(0) for a USB cam
    frame = camera.grab()                              # None if no frame yet
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Self

# Lower RTSP latency; must be set before the capture is opened.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)


class Camera:
    """Grabs frames from an RTSP stream or a local camera.

    Args:
        source: An RTSP URL string, or a camera index (0 = first USB camera).
        mirror: Flip the image left-right (some cameras are mirrored).
        reconnect_after: Seconds of silence before trying to reopen the stream.
    """

    def __init__(
        self,
        source: str | int = 0,
        mirror: bool = False,
        reconnect_after: float = 6.0,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.source = source
        self.mirror = mirror
        self.reconnect_after = reconnect_after

        self._capture: Any = None
        self._last_frame_time = time.monotonic()
        # grab() may run on a background thread while release() happens on
        # the main one — the lock keeps them from touching the capture at once.
        self._lock = threading.Lock()
        self._released = False
        self._open()

    def _make_capture(self) -> Any:
        cv2 = self._cv2

        if isinstance(self.source, int):
            capture = cv2.VideoCapture(self.source)
        else:
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _open(self) -> None:
        self._capture = self._make_capture()
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

    @property
    def opened(self) -> bool:
        """True while the stream is open (it may be False after a drop)."""
        return self._capture is not None and self._capture.isOpened()

    def grab(self) -> Any | None:
        """Get the newest frame, or None if the stream is temporarily dead.

        If the stream has been silent for ``reconnect_after`` seconds this
        automatically tries to open it again.
        """
        cv2 = self._cv2

        with self._lock:
            if self._released or self._capture is None:
                return None
            try:
                success, frame = self._capture.read()
            except (cv2.error, RuntimeError):
                # capture was released from another thread mid-read
                return None

        if not success or frame is None:
            quiet_for = time.monotonic() - self._last_frame_time
            if quiet_for >= self.reconnect_after:
                print(f"Stream lost, reconnecting to {self.source}...")
                try:
                    self._capture.release()
                except (cv2.error, RuntimeError):
                    pass  # already closed; _open() below starts fresh
                time.sleep(1.0)
                self._open()

            return None

        self._last_frame_time = time.monotonic()

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self) -> None:
        """Stop the camera and free its resources.

        Safe to call from any thread — it waits for an in-flight read()
        to finish first, so a background grabber can't crash on it.
        """
        with self._lock:
            self._released = True
            if self._capture is not None:
                try:
                    self._capture.release()
                except (self._cv2.error, RuntimeError):
                    pass  # already closed; nothing left to clean up
                self._capture = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
