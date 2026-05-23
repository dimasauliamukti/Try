import cv2
import onnxruntime as ort
import numpy as np
from exception import DetectionFailed, ModelLoadError
import logging

logger = logging.getLogger(__name__)


class DetectNeck:
    """
    Neck/shoulder bounding-box detector with a three-tier fallback strategy.

    Detection is attempted in the following order:
      1. ONNX model (YOLO-based, GPU-accelerated when available).
      2. Haar Cascade face detector — derives a neck region from the face bbox.
      3. Manual crop — returns the bottom half of the image as a last resort.

    Raises:
        ModelLoadError: If the ONNX session or Haar Cascade cannot be initialised.
        DetectionFailed: If all three detection methods fail to produce a result.
    """

    def __init__(self, model_path: str) -> None:
        """
        Load the ONNX inference session and the Haar Cascade face detector.

        Args:
            model_path: Path to the YOLO neck-detection ONNX model file.

        Raises:
            ModelLoadError: If the ONNX session fails to initialise, or if the
                            Haar Cascade XML file cannot be loaded.
        """
        try:
            self.onnx = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception as e:
            raise ModelLoadError(method="Bbox ONNX", cause=str(e))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.face_cascade.empty():
            raise ModelLoadError(method="Haar Cascade")

    def detect_onnx(self, image: np.ndarray, padding: int = 10) -> tuple:
        """
        Run the YOLO ONNX model to detect the neck bounding box.

        Pre-processes the image to a normalised (1, 3, H, W) float32 tensor,
        runs inference, selects the highest-confidence prediction, and applies
        symmetric padding to the resulting box.

        Args:
            image:   BGR input image (H x W x 3, uint8).
            padding: Number of pixels to expand each side of the predicted box.
                     Clamped to image boundaries. Defaults to 10.

        Returns:
            (crop, coords, score) where:
              - crop   (np.ndarray | None): Cropped BGR region containing the neck,
                                            or None if no detection was produced.
              - coords (tuple | None)     : (x1, y1, x2, y2) pixel coordinates of
                                            the padded box in the input image, or None.
              - score  (float)            : Confidence score of the best prediction.
        """
        inp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]  # (1, 3, 640, 640)
        output_raw = self.onnx.run(None, {self.onnx.get_inputs()[0].name: inp})[0]
        output = output_raw[0].T  # (8400, 5)
        if len(output) == 0:
            return None, None, 0.0

        best = output[output[:, 4].argmax()]
        cx, cy, w, h, score = best
        ih, iw = image.shape[:2]
        x1 = max(0, int(cx - w / 2) - padding)
        y1 = max(0, int(cy - h / 2) - padding)
        x2 = min(iw, int(cx + w / 2) + padding)
        y2 = min(ih, int(cy + h / 2) + padding)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2), float(score)

    def detect_cascade(self, faces: np.ndarray, image: np.ndarray) -> tuple:
        """
        Derive a neck crop from the largest face detected by the Haar Cascade.

        Selects the largest face rectangle by area, then estimates the neck
        region as a square patch immediately below the chin whose width equals
        the face width.

        Args:
            faces: Array of face rectangles returned by
                   ``cv2.CascadeClassifier.detectMultiScale`` (N x 4).
                   Pass an empty array when no faces were found.
            image: BGR input image (H x W x 3, uint8).

        Returns:
            (crop, coords, score) where:
              - crop   (np.ndarray | None): Cropped BGR neck region, or None if
                                            no faces were found or the derived
                                            region is empty.
              - coords (tuple | None)     : (x1, y1, x2, y2) pixel coordinates of
                                            the crop in the input image, or None.
              - score  (float)            : Always 0.0 (no confidence from Haar).
        """
        if len(faces) == 0:
            return None, None, 0.0
        h, w = image.shape[:2]
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        x_min = max(0, fx)
        x_max = min(w, fx + fw)
        ratio = int((x_max - x_min))

        y_min = max(0, fy + fh - 120)
        y_max = min(h, y_min + ratio)

        below_face = image[y_min:y_max, 0:w]
        if below_face.size == 0:
            return None, None, 0.0

        return below_face, (0, y_min, w, y_max), 0.0

    def detect_manual(self, image: np.ndarray) -> tuple:
        """
        Fallback crop that returns the bottom half of the image as the neck region.

        Used when both the ONNX model and the Haar Cascade fail to produce a
        valid bounding box. Guarantees a non-empty crop as long as the image
        has at least one row.

        Args:
            image: BGR input image (H x W x 3, uint8).

        Returns:
            (crop, coords, score) where:
              - crop   (np.ndarray): Bottom half of the input image.
              - coords (tuple)     : (0, h//2, w, h) pixel coordinates.
              - score  (float)     : Always 0.0 (heuristic, no confidence).
        """
        h, w = image.shape[:2]
        height_crop = int(h / 2)
        x_min = 0
        x_max = w
        y_min = height_crop
        y_max = h
        return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max), 0.0

    def detect_bbox(self, image: np.ndarray) -> tuple:
        """
        Detect the neck bounding box using a three-tier fallback strategy.

        Attempts detection in the following order, falling back to the next
        method when the current one fails or returns a low-confidence result:

          1. **ONNX** — YOLO model; skipped if score is in (0, 0.5).
          2. **Haar Cascade** — face-based neck estimation.
          3. **Manual** — bottom-half crop.

        Args:
            image: BGR input image (H x W x 3, uint8).

        Raises:
            DetectionFailed: If all three methods fail to produce a valid crop.

        Returns:
            (bbox, coordinate, score) where:
              - bbox       (np.ndarray): Cropped BGR region containing the neck.
              - coordinate (tuple)     : (x1, y1, x2, y2) pixel coordinates of
                                         the crop in the input image.
              - score      (float)     : Confidence score from the ONNX model,
                                         or 0.0 when a fallback method was used.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        bbox, coordinate, score = self.detect_onnx(image)

        if 0 < score < 0.5:
            print("ONNX bbox score too low, Fallback To Haar Cascade")
            logger.warning("ONNX bbox score too low, Fallback To Haar Cascade")
            bbox, coordinate, score = self.detect_cascade(faces, image)

        if bbox is None:
            print("ONNX Failed, Fallback To Haar Cascade")
            logger.warning("ONNX Failed, Fallback To Haar Cascade")
            bbox, coordinate, score = self.detect_cascade(faces, image)
        if bbox is None:
            print("Haar Cascade Failed, Fallback To Manual")
            logger.warning("Haar Cascade Failed, Fallback To Manual")
            bbox, coordinate, score = self.detect_manual(image)
        if bbox is None:
            raise DetectionFailed(method="Neck Detection", cause="-")
        return bbox, coordinate, score