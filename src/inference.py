"""
test_pipeline.py
End-to-end debug/test script for the neck-shoulder detection pipeline.

Runs the full inference pipeline on a single image and displays intermediate
and final results using OpenCV windows. Intended for local development and
visual inspection only - not for production use.

Pipeline stages:
  1. Load all ML models (MediaPipe, SegRefiner, YOLO bbox, HRNet keypoints).
  2. Read and validate the test image.
  3. Crop the upper-body region and resize to the model input resolution.
  4. Generate a refined skin/neck segmentation mask.
  5. Check shoulder visibility via hair-region analysis.
  6. Detect the neck bounding box.
  7. Detect and snap neck/shoulder keypoints to the mask contour.
  8. Map keypoints back to the original image coordinate space.
  9. Serialize results to a JSON-compatible dict.
  10. Display annotated images and write result images to disk.
"""

from bbox import DetectNeck
from config import BASE_DIR, DATASET_DIR, MODEL_DIR, SEGREFINER_DIR
from segmentation_mp import MediaPipeSkinSegmenter
from segrefiner import SegRefinerWrapper
import cv2
from resize_image import resize, crop_upper_body, map_keypoint_to_original
from keypoints import DetectKeypoints
import time
from preprocess import snap_to_nearest_contour, boundary_blend, mask_to_coords, to_json, hair_detection
import logging
from exception import ModelLoadError, DetectionFailed, ShoulderNeckNotVisible
import torch
import numpy as np

# Display buffers — populated during the pipeline and consumed in `finally`
cropped          = None  # Bbox crop with raw keypoints drawn on it
display_resized  = None  # 640x640 resized image with snapped keypoints
display_original = None  # Full-resolution image with mapped keypoints

# Logger setup (standalone; does not inherit root logger handlers)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)


start = time.time()
try:
    # Stage 1 - Model loading
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MediaPipe multi-class segmenter (skin / hair / body)
    segmenter = MediaPipeSkinSegmenter("../models/selfie_multiclass.tflite")

    # SegRefiner - high-resolution boundary refinement
    refiner = SegRefinerWrapper(
        config_path     = "../SegRefiner/configs/segrefiner/segrefiner_hr.py",
        checkpoint_path = "../models/segrefiner_hr_latest.pth",
        device          = device,
        target_size     = 640
    )

    # YOLO-based neck bounding-box detector
    bbox_model = DetectNeck("../models/new_bbox3/weights/best.onnx")

    # HRNet-based neck/shoulder keypoint detector
    keypoints_model = DetectKeypoints("../keypoints_model/models/necklace_keypoints.onnx")

    # Stage 2 - Image loading and validation
    img = cv2.imread(str(DATASET_DIR / "Test/b.jpg"))
    if img is None:
        raise ValueError("Format Image Not Supported")
    h, w = img.shape[:2]

    display_original = img.copy()

    # Stage 3 - Upper-body crop and resize
    # face_info stores the crop offset/scale needed to map coordinates back later
    face, face_info = crop_upper_body(img)
    image, r, top, left = resize(face)
    display_resized = image.copy()

    # Stage 4 - Segmentation and mask refinement
    # MediaPipe produces coarse masks; SegRefiner sharpens boundaries;
    # boundary_blend fuses the two for the final per-region masks.
    mask, image_seg, skin_mask, neck_face, neck_body = segmenter.segment(image)

    mask_segrefiner = refiner.run(image, mask, iterations=1)
    mask_final     = boundary_blend(mask,      mask_segrefiner, boundary_width=3)
    neckface_final = boundary_blend(neck_face, mask_segrefiner, boundary_width=3)
    neckbody_final = boundary_blend(neck_body, mask_segrefiner, boundary_width=3)

    print("neckbody_final unique:", np.unique(neckbody_final))  # Should contain values > 0
    print("neckbody shape:", neckbody_final.shape)

    mask_coordinate = mask_to_coords(neckbody_final)

    # Stage 5 - Shoulder visibility check via hair-region ratio
    # If the ratio is too low, hair is likely occluding the shoulder area
    # and keypoints cannot be reliably detected.
    shoulder_image_ratio = hair_detection(mask_coordinate, left)
    print(shoulder_image_ratio)
    if shoulder_image_ratio < 0.4:
        raise ShoulderNeckNotVisible(cause="Hair may be covering the shoulder area")

    # Stage 6 - Neck bounding-box detection (YOLO with cascade fallback)
    bbox, bbox_coords, score = bbox_model.detect_bbox(image)
    print(score)
    cropped = bbox.copy()
    bx1, by1, bx2, by2 = bbox_coords

    # Stage 7 - Keypoint detection and contour snapping
    # Raw keypoints are detected inside the bbox crop, then offset back
    # to the full 640x640 coordinate space before snapping to the nearest
    # point on the neck-face mask contour.
    keypoints = keypoints_model.detect_kp(bbox)

    # Draw raw keypoints on the bbox crop for debugging
    for (x, y, score) in keypoints:
        cv2.circle(cropped, (x, y), 1, (0, 255, 0), -1)
        print(x, y)

    # Offset keypoints from bbox-local to full-resized coordinates
    keypoints_on_resized = [(int(x) + bx1, int(y) + by1, float(score)) for x, y, score in keypoints]

    # Snap each keypoint to the nearest contour point on the neck mask
    keypoints_snapped = snap_to_nearest_contour(neckface_final, mask_final, keypoints_on_resized)
    print(keypoints_snapped)

    rx, ry, rscore, rval = keypoints_snapped[0]
    lx, ly, lscore, lval = keypoints_snapped[1]

    if not rval or not lval:
        logger.warning("Snap Failed, Using Raw Keypoints")

    # Low keypoint score indicates the neck/shoulder is not clearly visible
    if rscore < 0.35 or lscore < 0.35:
        raise ShoulderNeckNotVisible(cause="Neck-shoulder area not clearly visible")

    # Stage 8 - Map keypoints back to original image coordinates
    keypoints_original = [
        map_keypoint_to_original(x, y, r, top, left, face_info)
        for x, y, _, _ in keypoints_snapped
    ]
    keypoints_original_no_snapped = [
        map_keypoint_to_original(x, y, r, top, left, face_info)
        for x, y, _ in keypoints_on_resized
    ]
    print(keypoints_original_no_snapped)

    # Draw snapped keypoints on the resized display image
    for q, w, _, _ in keypoints_snapped:
        cv2.circle(display_resized, (q, w), 5, (255, 100, 0), -1)

    # Draw mapped keypoints on the full-resolution display image
    for x_snap, y_snap in keypoints_original:
        cv2.circle(display_original, (x_snap, y_snap), 1, (255, 100, 0), -1)
        print(x_snap, y_snap)

    # Stage 9 - Serialize results to JSON-compatible dict
    output_json = {
        "right_neck_shoulder_point": {
            "original_point": to_json(list(keypoints_original[0])),
            "resized_point" : to_json([rx, ry]),
            "score"         : to_json(rscore),
        },
        "left_neck_shoulder_point": {
            "original_point": to_json(list(keypoints_original[1])),
            "resized_point" : to_json([lx, ly]),
            "score"         : to_json(lscore),
        },
        "mask_area": to_json(mask_coordinate),
        "img_size" : to_json([w, h]),
    }

except ModelLoadError as e:
    print(f"[ERROR] {e}")
except DetectionFailed as e:
    print(f"[ERROR] {e}")
except Exception as e:
    print(f"[ERROR] {e}")

finally:
    # Stage 10 - Display annotated results and write output images to disk
    # Runs regardless of success or failure so windows always appear and
    # partial results are preserved for post-mortem inspection.
    end = time.time()
    print(f"Total waktu: {end - start:.2f} detik")

    if cropped is not None:
        cv2.imshow("Keypoints", cropped)

    if display_resized is not None:
        cv2.imshow("Keypoints (640x640)", display_resized)
        cv2.imwrite("Result_640.jpg", display_resized)

    if display_original is not None:
        cv2.imshow("Keypoint Original", display_original)
        cv2.imwrite("Result_Fix.jpg", display_original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()