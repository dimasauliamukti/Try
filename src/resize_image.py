import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


def crop_upper_body(image: np.ndarray, padding_top: float = 0.1,
                    padding_side: float = 0.1, body_ratio: float = 0.5) -> tuple:
    """
    Detect the largest face and crop the upper-body region around it.

    Uses Haar cascade face detection to locate the largest face in the frame.
    Padding and body extension are applied as fractions of the detected face
    dimensions. If no face is found, falls back to cropping the image with a
    10 % margin on all sides and logs an info-level warning.

    Args:
        image:        Input BGR image array (H x W x 3, uint8).
        padding_top:  Extra space above the face as a fraction of face height
                      (default: 0.1).
        padding_side: Extra space on each side as a fraction of face width
                      (default: 0.1).
        body_ratio:   How far below the face chin to extend the crop, as a
                      fraction of face height (default: 0.5).

    Returns:
        A two-element tuple ``(cropped, crop_info)`` where:
          - cropped   (np.ndarray): Cropped BGR image of the upper-body region.
          - crop_info (dict): Crop boundary coordinates in the original image
                              with keys ``'x1'``, ``'y1'``, ``'x2'``, ``'y2'``.
    """
    h, w = image.shape[:2]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        logger.info("Crop Upper Body : Face not Detected")
        margin_x  = int(w * 0.10)
        margin_y  = int(h * 0.10)
        crop_info = {'x1': margin_x, 'y1': margin_y,
                     'x2': w - margin_x, 'y2': h - margin_y}
        return image[margin_y:h - margin_y, margin_x:w - margin_x], crop_info

    logger.info("Crop Upper Body : Face Detected")

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    pad_top     = int(fh * padding_top)
    pad_side    = int(fw * padding_side)
    body_height = int(fh * body_ratio)

    x1 = max(0, fx - pad_side)
    y1 = max(0, fy - pad_top)
    x2 = min(w, fx + fw + pad_side)
    y2 = min(h, fy + fh + body_height)

    cropped   = image[y1:y2, x1:x2]
    crop_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    return cropped, crop_info


def resize(image: np.ndarray, target_size: tuple = (640, 640),
           color: tuple = (0, 0, 0)) -> tuple:
    """
    Resize an image to ``target_size`` using letterboxing to preserve aspect ratio.

    The image is scaled uniformly by the smaller of the width and height
    ratios, then centered on a solid-color canvas of the requested dimensions.
    If the image already matches ``target_size`` exactly, it is returned
    unchanged with a scale ratio of 1.0 and zero padding.

    Args:
        image:       Input BGR image array (H x W x 3, uint8).
        target_size: ``(height, width)`` of the output canvas (default: (640, 640)).
        color:       Background fill color for the padding areas in BGR
                     (default: (0, 0, 0) — black).

    Returns:
        A four-element tuple ``(canvas, r, top, left)`` where:
          - canvas (np.ndarray): Letterboxed image of shape ``target_size``.
          - r      (float):      Uniform scale ratio applied
                                 (resized dimension / original dimension).
          - top    (int):        Vertical padding in pixels (top margin).
          - left   (int):        Horizontal padding in pixels (left margin).
    """
    h, w             = image.shape[:2]
    target_h, target_w = target_size

    if (h, w) == (target_h, target_w):
        return image, 1.0, 0, 0

    r                    = min(target_w / w, target_h / h)
    resized_w, resized_h = int(w * r), int(h * r)

    resized = cv2.resize(image, (resized_w, resized_h))
    canvas  = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    top  = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2

    canvas[top:top + resized_h, left:left + resized_w] = resized

    return canvas, r, top, left


def map_keypoint_to_original(kp_x: int, kp_y: int, r: float,
                              top: int, left: int, crop_info: dict) -> tuple:
    """
    Map a keypoint coordinate from letterbox space back to the original image.

    Applies the inverse of the two-stage transformation pipeline used during
    preprocessing:

    .. code-block:: text

        letterbox space  →  crop image space  →  original image space

    Step 1 reverses the letterbox padding and scale applied by ``resize``.
    Step 2 offsets the result by the crop origin recorded in ``crop_info``
    from ``crop_upper_body``.

    Args:
        kp_x:      Keypoint x-coordinate in letterbox (canvas) space (pixels).
        kp_y:      Keypoint y-coordinate in letterbox (canvas) space (pixels).
        r:         Scale ratio returned by ``resize``
                   (resized dimension / original dimension).
        top:       Vertical padding in pixels returned by ``resize``.
        left:      Horizontal padding in pixels returned by ``resize``.
        crop_info: Dict with keys ``'x1'`` and ``'y1'`` specifying the
                   top-left corner of the crop in the original image,
                   as returned by ``crop_upper_body``.

    Returns:
        A two-element tuple ``(x_orig, y_orig)`` containing the keypoint
        coordinates mapped to the original image space (pixels, int).
    """
    # Step 1: Letterbox space → crop image space
    x_crop = int((kp_x - left) / r)
    y_crop = int((kp_y - top)  / r)

    # Step 2: Crop image space → original image space
    x_orig = x_crop + crop_info['x1']
    y_orig = y_crop + crop_info['y1']

    return x_orig, y_orig