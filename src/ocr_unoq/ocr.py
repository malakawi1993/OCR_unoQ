"""The OCR engine — find and read text in images.

For kids: just use ``read_text(image)`` and you get a list of words!

    from ocr_unoq import read_text

    results = read_text("photo.jpg")
    for result in results:
        print(result.text, result.confidence)

Everything else (models, thresholds) has sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only for type checkers, not imported at runtime
    import cv2
    import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"

DEFAULT_DET_MODEL = MODELS_DIR / "detection" / "v4" / "det.onnx"
DEFAULT_REC_MODEL = MODELS_DIR / "languages" / "english" / "v4" / "rec.onnx"
DEFAULT_DICT = MODELS_DIR / "languages" / "english" / "v4" / "dict.txt"

# Defaults tuned for a small SBC (Arduino UNO Q) with an RTSP camera.
DEFAULT_MIN_CONFIDENCE = 0.40
DEFAULT_BOX_THRESH = 0.5
DEFAULT_MAX_WIDTH = 640
# Limit the ONNX runtime's threads so OCR leaves CPU for the camera feed.
# -1 = let onnxruntime decide (all cores) — fine on a desktop, too greedy
# on a 4-core A53 board where the feed also needs to run.
DEFAULT_INTRA_OP_THREADS = 2
DEFAULT_INTER_OP_THREADS = 1


@dataclass
class TextResult:
    """One piece of text that was found in an image."""

    text: str
    confidence: float
    box: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        """Middle of the text box — handy for pointing a robot at it."""
        if not self.box:
            return (0, 0)
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    @property
    def top_left(self) -> tuple[int, int]:
        """Top-left corner of the text box."""
        if not self.box:
            return (0, 0)
        return (min(p[0] for p in self.box), min(p[1] for p in self.box))

    def __str__(self) -> str:
        return f"{self.text} ({self.confidence:.0%})"


class TextReader:
    """Reads text out of images. Create one, then call it as many times as you like.

    Example::

        reader = TextReader()
        results = reader("photo.jpg")          # or a camera frame (numpy array)
        print(results[0].text if results else "no text")
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        box_thresh: float = DEFAULT_BOX_THRESH,
        max_width: int = DEFAULT_MAX_WIDTH,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        dict_path: str | Path | None = None,
        intra_op_threads: int = DEFAULT_INTRA_OP_THREADS,
        inter_op_threads: int = DEFAULT_INTER_OP_THREADS,
    ) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.min_confidence = min_confidence
        self.box_thresh = box_thresh
        self.max_width = max_width

        det = Path(det_model_path or DEFAULT_DET_MODEL)
        rec = Path(rec_model_path or DEFAULT_REC_MODEL)
        dictionary = Path(dict_path or DEFAULT_DICT)

        missing = [str(p) for p in (det, rec, dictionary) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing OCR model file(s):\n  "
                + "\n  ".join(missing)
                + "\nRun `python download_models.py` first."
            )

        self._ocr = RapidOCR(
            det_model_path=str(det),
            rec_model_path=str(rec),
            rec_keys_path=str(dictionary),
            intra_op_num_threads=intra_op_threads,
            inter_op_num_threads=inter_op_threads,
        )

    def __call__(self, image: Any) -> list[TextResult]:
        """Run OCR on an image (a file path or a camera frame)."""
        import cv2

        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))
            if loaded is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            image = loaded

        height, width = image.shape[:2]
        scale = min(1.0, self.max_width / width)

        if scale < 1.0:
            ocr_image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            ocr_image = image

        results, _ = self._ocr(
            ocr_image,
            text_score=self.min_confidence,
            box_thresh=self.box_thresh,
        )

        found: list[TextResult] = []
        if not results:
            return found

        for box, text, score in results:
            text = str(text).strip()
            confidence = float(score)

            if not text or confidence < self.min_confidence:
                continue

            full_box = [
                (int(float(p[0]) / scale), int(float(p[1]) / scale)) for p in box
            ]
            found.append(TextResult(text=text, confidence=confidence, box=full_box))

        return found


# A shared reader so `read_text()` works with zero setup.
_default_reader: TextReader | None = None


def get_reader() -> TextReader:
    """Return the shared TextReader (created on first use)."""
    global _default_reader
    if _default_reader is None:
        _default_reader = TextReader()
    return _default_reader


def read_text(image: Any) -> list[TextResult]:
    """Read all text in an image. The one function kids should know.

    Args:
        image: A file path ("photo.jpg") or a camera frame (numpy array).

    Returns:
        A list of TextResult — empty if no text was found.
    """
    return get_reader()(image)


def draw_results(
    image: Any,
    results: list[TextResult],
    color: tuple[int, int, int] = (0, 255, 0),
) -> Any:
    """Draw boxes and labels for OCR results. Returns a NEW image (your frame is untouched)."""
    import cv2

    canvas = image.copy() if hasattr(image, "copy") else image

    for result in results:
        points = result.box
        if len(points) < 3:
            continue

        for index in range(len(points)):
            cv2.line(
                canvas,
                points[index],
                points[(index + 1) % len(points)],
                color,
                2,
            )

        label = f"{result.text} {result.confidence:.0%}"
        x, y = result.top_left
        cv2.putText(
            canvas,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return canvas


def status_line(results: list[TextResult], elapsed_ms: float) -> str:
    """A short human-readable summary, e.g. for a HUD or the console."""
    if results:
        state = f"{len(results)} text region(s)"
    else:
        state = "no text found"
    return f"OCR: {state} ({elapsed_ms:.0f} ms)"
