"""
augmentation_pipeline.py
Augments a dataset of person images by overlaying four collar/shirt styles
(turtleneck, v-neck, crewneck, collar shirt) onto the detected torso region.

Output format: COCO-format JSON with optional YOLO bounding-box crop.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config_keypoints import BASE_DIR, DATASET_DIR, MODEL_BBOX_DIR
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from segmentation_mp import MediaPipeSkinSegmenter
from segrefiner import SegRefinerWrapper

STYLE_COLOR_MAP = {
    "turtleneck"  : (30,  30,  30),
    "vneck"       : (0,   0,  200),
    "crewneck"    : (0,  128,   0),
    "collar_shirt": (42,  82, 139),
}


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def boundary_blend(
    mask_mediapipe: np.ndarray,
    mask_segrefiner: np.ndarray,
    boundary_width: int = 15,
) -> np.ndarray:
    """
    Blend two binary masks by replacing the boundary zone of the MediaPipe
    mask with values from the SegRefiner mask, while keeping the interior
    fully white.

    Args:
        mask_mediapipe:  Binary mask from MediaPipe (uint8, 0/255).
        mask_segrefiner: Binary mask from SegRefiner  (uint8, 0/255).
        boundary_width:  Half-width of the erosion kernel in pixels.

    Returns:
        Blended binary mask (uint8, 0/255).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_width * 2 + 1, boundary_width * 2 + 1),
    )
    eroded        = cv2.erode(mask_mediapipe, kernel)
    boundary_zone = cv2.subtract(mask_mediapipe, eroded)

    mask_combined = mask_mediapipe.copy()
    mask_combined[boundary_zone > 0] = mask_segrefiner[boundary_zone > 0]
    mask_combined[eroded       > 0] = 255
    return mask_combined


def generate_mask(image: np.ndarray, segmenter, refiner) -> np.ndarray:
    """
    Generate a refined body/torso mask for a BGR image.

    Steps:
      1. Run MediaPipe segmenter to get a coarse mask.
      2. Run SegRefiner to sharpen mask edges.
      3. Blend the two masks at the boundary zone.

    Args:
        image:     BGR image (H x W x 3, uint8).
        segmenter: MediaPipeSkinSegmenter instance.
        refiner:   SegRefinerWrapper instance.

    Returns:
        Refined binary mask (uint8, 0/255).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask, _, _, _, neck_mask = segmenter.segment(image_rgb)

    mask_segrefiner = refiner.run(image, mask, iterations=1)
    return boundary_blend(neck_mask, mask_segrefiner, boundary_width=3)

def get_collar_styles(mask_bin: np.ndarray, image: np.ndarray) -> tuple[list[dict], int | None]:
    """
    Build a list of shirt-style parameter dictionaries based on the detected
    torso mask.

    Args:
        mask_bin: Binary torso mask (uint8, 0/255).
        image:    Original BGR image.

    Returns:
        (styles, top_of_mask) where:
          - styles       : list of style dicts (name, y_start, cutout, color, patch_h).
          - top_of_mask  : first row index where the mask is non-zero, or None.
    """
    h, w = image.shape[:2]

    mask_rows = np.where(mask_bin.max(axis=1) > 0)[0]
    if len(mask_rows) == 0:
        return [], None

    top_of_mask = mask_rows[0]
    patch_h     = h - top_of_mask

    styles = [
        {
            "name"   : "turtleneck",
            "y_start": max(0, top_of_mask - int(patch_h * 0.4)),
            "cutout" : None,
            "color"  : STYLE_COLOR_MAP["turtleneck"],
            "patch_h": patch_h,
        },
        {
            "name"   : "vneck",
            "y_start": top_of_mask,
            "cutout" : {
                "type" : "v",
                "depth": int(patch_h * 0.5),
                "width": int(w * 0.25),
            },
            "color"  : STYLE_COLOR_MAP["vneck"],
            "patch_h": patch_h,
        },
        {
            "name"   : "crewneck",
            "y_start": top_of_mask,
            "cutout" : {
                "type"  : "crew",
                "height": int(patch_h * 0.15),
                "width" : int(w * 0.18),
            },
            "color"  : STYLE_COLOR_MAP["crewneck"],
            "patch_h": patch_h,
        },
        {
            "name"   : "collar_shirt",
            "y_start": top_of_mask,
            "cutout" : None,
            "color"  : STYLE_COLOR_MAP["collar_shirt"],
            "patch_h": patch_h,
        },
    ]

    return styles, top_of_mask

def find_left_contour_tip(mask_bin: np.ndarray, cx: int, top_row: int, collar_bottom_y: int) -> int:
    """Return the leftmost column of the mask in the region left of cx."""
    roi  = mask_bin[top_row:collar_bottom_y + 1, :cx]
    cols = np.where(roi.max(axis=0) > 0)[0]
    return int(cols[0]) if len(cols) > 0 else 0


def find_right_contour_tip(mask_bin: np.ndarray, cx: int, top_row: int, collar_bottom_y: int, w: int) -> int:
    """Return the rightmost column of the mask in the region right of cx."""
    roi  = mask_bin[top_row:collar_bottom_y + 1, cx:]
    cols = np.where(roi.max(axis=0) > 0)[0]
    return cx + int(cols[-1]) if len(cols) > 0 else w - 1


def compute_left_triangle(
    mask_bin: np.ndarray,
    cx: int,
    overall_top_row: int,
    collar_bottom_y: int,
) -> np.ndarray:
    """
    Compute the three vertices of the left collar-flap triangle.

    Args:
        mask_bin:        Binary torso mask.
        cx:              Horizontal centre of the image.
        overall_top_row: First row of the mask.
        collar_bottom_y: Bottom row of the collar zone.

    Returns:
        (3, 2) int32 array of triangle vertices.
    """
    left_area = mask_bin[:, :cx]
    left_rows = np.where(left_area.max(axis=1) > 0)[0]

    if len(left_rows) == 0:
        top_row_left    = overall_top_row
        right_edge_left = cx - 1
    else:
        top_row_left    = min(int(left_rows[0]), collar_bottom_y - 1)
        left_top_cols   = np.where(left_area[top_row_left] > 0)[0]
        right_edge_left = int(left_top_cols[-1]) if len(left_top_cols) > 0 else cx - 1

    far_left_x = find_left_contour_tip(mask_bin, cx, top_row_left, collar_bottom_y)
    return np.array(
        [[right_edge_left, top_row_left],
         [right_edge_left, collar_bottom_y],
         [far_left_x,      collar_bottom_y]],
        dtype=np.int32,
    )


def compute_right_triangle(
    mask_bin: np.ndarray,
    cx: int,
    overall_top_row: int,
    collar_bottom_y: int,
    w: int,
) -> np.ndarray:
    """
    Compute the three vertices of the right collar-flap triangle.

    Args:
        mask_bin:        Binary torso mask.
        cx:              Horizontal centre of the image.
        overall_top_row: First row of the mask.
        collar_bottom_y: Bottom row of the collar zone.
        w:               Image width.

    Returns:
        (3, 2) int32 array of triangle vertices.
    """
    right_area = mask_bin[:, cx:]
    right_rows = np.where(right_area.max(axis=1) > 0)[0]

    if len(right_rows) == 0:
        top_row_right   = overall_top_row
        left_edge_right = cx
    else:
        top_row_right   = min(int(right_rows[0]), collar_bottom_y - 1)
        right_top_cols  = np.where(right_area[top_row_right] > 0)[0]
        left_edge_right = cx + int(right_top_cols[0]) if len(right_top_cols) > 0 else cx

    far_right_x = find_right_contour_tip(mask_bin, cx, top_row_right, collar_bottom_y, w)
    return np.array(
        [[left_edge_right, top_row_right],
         [left_edge_right, collar_bottom_y],
         [far_right_x,     collar_bottom_y]],
        dtype=np.int32,
    )

def blend_color_to_mask(image: np.ndarray, side_mask: np.ndarray, color: tuple) -> np.ndarray:
    """
    Composite a solid color (with slight noise) onto the image wherever
    side_mask is non-zero, using a Gaussian-blurred soft edge.

    Args:
        image:     BGR source image.
        side_mask: Single-channel mask (uint8, 0/255).
        color:     BGR color tuple to fill.

    Returns:
        Composited BGR image (uint8).
    """
    h, w  = image.shape[:2]
    noise = np.random.normal(0, 8, (h, w, 3))
    color_layer = np.clip(
        np.full((h, w, 3), color, dtype=np.float32) + noise, 0, 255
    ).astype(np.uint8)

    alpha = cv2.GaussianBlur(side_mask.astype(np.float32), (15, 15), 0) / 255.0
    alpha = alpha[:, :, np.newaxis]

    return (color_layer * alpha + image * (1 - alpha)).astype(np.uint8)


def apply_shirt_style(image: np.ndarray, style: dict, mask_bin: np.ndarray) -> np.ndarray:
    """
    Overlay a shirt-style color onto the image according to style.

    Supported styles:
      - "turtleneck"   : Solid block from above the torso downward.
      - "vneck"        : Torso block with a triangular V cut-out at the collar.
      - "crewneck"     : Torso block with a small elliptical cut-out at the collar.
      - "collar_shirt" : Two triangular collar flaps on the left and right.

    Args:
        image:    BGR source image.
        style:    Style dict from get_collar_styles.
        mask_bin: Binary torso mask (uint8, 0/255).

    Returns:
        Augmented BGR image (uint8).
    """
    h, w    = image.shape[:2]
    color   = style["color"]
    patch_h = style["patch_h"]
    name    = style["name"]

    if name == "collar_shirt":
        mask_rows = np.where(mask_bin.max(axis=1) > 0)[0]
        if len(mask_rows) == 0:
            return image

        cx              = w // 2
        overall_top_row = int(mask_rows[0])
        collar_bottom_y = min(h - 1, overall_top_row + int(patch_h * 0.28))

        pts_left  = compute_left_triangle(mask_bin, cx, overall_top_row, collar_bottom_y)
        pts_right = compute_right_triangle(mask_bin, cx, overall_top_row, collar_bottom_y, w)

        collar_mask = mask_bin.copy()
        cv2.fillPoly(collar_mask, [pts_left],  255)
        cv2.fillPoly(collar_mask, [pts_right], 255)

        return blend_color_to_mask(image, collar_mask, color)

    y_start    = style["y_start"]
    shirt_mask = np.zeros((h, w), dtype=np.uint8)
    shirt_mask[y_start:h, :] = 255

    if name == "vneck":
        v_depth = style["cutout"]["depth"]
        v_width = style["cutout"]["width"]
        cx      = w // 2
        for i in range(v_depth):
            gap_half = int(v_width * (1 - i / v_depth) / 2)
            y        = y_start + i
            if gap_half > 0 and y < h:
                shirt_mask[y, cx - gap_half:cx + gap_half] = 0

    elif name == "crewneck":
        crew_h = style["cutout"]["height"]
        crew_w = style["cutout"]["width"]
        cx     = w // 2
        for i in range(crew_h):
            gap_half = int(crew_w * (1 - i / crew_h) / 2)
            y        = y_start + i
            if gap_half > 0 and y < h:
                shirt_mask[y, cx - gap_half:cx + gap_half] = 0

    shirt_mask = cv2.bitwise_and(shirt_mask, mask_bin)
    return blend_color_to_mask(image, shirt_mask, color)



def run_yolo(
    model: YOLO,
    image: np.ndarray,
    padding: int = 10,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray | None, tuple | None]:
    """
    Run YOLO inference via Ultralytics and return the crop of the
    highest-confidence detection.

    Args:
        model:           Ultralytics YOLO model instance.
        image:           BGR input image (H x W x 3, uint8).
        padding:         Pixels to expand each side of the predicted box.
        conf_threshold:  Minimum confidence score to accept a detection.

    Returns:
        (cropped_image, (x1, y1, x2, y2)) or (None, None) if no detection.
    """
    ih, iw = image.shape[:2]

    results = model(image, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()

    valid = confs >= conf_threshold
    if not valid.any():
        return None, None

    best_idx = int(confs[valid].argmax())
    xyxy     = boxes.xyxy.cpu().numpy()[valid][best_idx]

    x1 = max(0,  int(xyxy[0]) - padding)
    y1 = max(0,  int(xyxy[1]) - padding)
    x2 = min(iw, int(xyxy[2]) + padding)
    y2 = min(ih, int(xyxy[3]) + padding)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

def duplicate_annotations_json(ann_file: str, out_file: str, new_entries: list[dict]) -> None:
    """
    Append new image/annotation records to a COCO JSON file.

    For every entry in new_entries, the function copies all annotation
    records of the original image (identified by stem) and attaches them
    to the new image entry with updated IDs.

    Args:
        ann_file:    Path to the original COCO JSON annotation file.
        out_file:    Path to write the augmented COCO JSON.
        new_entries: List of dicts with keys:
                       - orig_stem    (str): stem of the original image filename.
                       - new_filename (str): filename of the augmented image.
                       - img_width    (int): width  of the augmented image.
                       - img_height   (int): height of the augmented image.
    """
    with open(ann_file) as f:
        coco = json.load(f)

    stem_to_img = {Path(img["file_name"]).stem: img for img in coco["images"]}
    id_to_anns  = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_images      = list(coco["images"])
    new_annotations = list(coco["annotations"])
    new_img_id      = max(img["id"] for img in coco["images"]) + 1
    new_ann_id      = max(ann["id"] for ann in coco["annotations"]) + 1

    success_count = skip_count = 0
    for entry in new_entries:
        orig_stem = entry["orig_stem"]
        if orig_stem not in stem_to_img:
            skip_count += 1
            continue

        img_w, img_h = entry["img_width"], entry["img_height"]
        new_images.append({
            "id"       : new_img_id,
            "file_name": entry["new_filename"],
            "width"    : img_w,
            "height"   : img_h,
        })

        for ann in id_to_anns.get(stem_to_img[orig_stem]["id"], []):
            new_annotations.append({
                "id"           : new_ann_id,
                "image_id"     : new_img_id,
                "category_id"  : ann["category_id"],
                "segmentation" : [],
                "area"         : float(img_w * img_h),
                "bbox"         : [0.0, 0.0, float(img_w), float(img_h)],
                "keypoints"    : list(ann["keypoints"]),
                "num_keypoints": ann["num_keypoints"],
                "iscrowd"      : 0,
            })
            new_ann_id += 1

        new_img_id += 1
        success_count += 1

    coco["images"]      = new_images
    coco["annotations"] = new_annotations

    with open(out_file, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"JSON saved        : {out_file}")
    print(f"Total images      : {len(new_images)}")
    print(f"Total annotations : {len(new_annotations)}")
    print(f"Succeeded         : {success_count}")
    print(f"Skipped           : {skip_count}")


def run_augmentation_pipeline(
    image_dir: str,
    output_full_dir: str,
    segmenter,
    refiner,
    ann_file: str,
    out_ann_file: str,
    output_crop_dir: str,
    yolo_model_path: str,
    conf_threshold: float = 0.25,
    padding: int = 10,
) -> None:
    """
    Run the full shirt-style augmentation pipeline on a directory of images.

    For each input image, the function first verifies that a corresponding
    annotation entry exists in the COCO JSON file. Images without a matching
    annotation are skipped entirely — no augmented or cropped files are written
    to disk. For valid images, four augmented variants (turtleneck, v-neck,
    crewneck, collar shirt) are generated and saved. Annotation files are
    updated to include the new images in COCO JSON format.

    Args:
        image_dir:       Directory of input images (.jpg / .png).
        output_full_dir: Directory to write full-size augmented images.
        segmenter:       MediaPipeSkinSegmenter instance.
        refiner:         SegRefinerWrapper instance.
        ann_file:        Path to original COCO JSON annotation file.
        out_ann_file:    Path to write augmented COCO JSON.
        output_crop_dir: Directory to write YOLO-cropped images.
        yolo_model_path: Path to YOLO weights (.pt or .onnx).
        conf_threshold:  Minimum confidence to accept a YOLO detection.
        padding:         Pixels to pad around each detected bounding box.
    """
    image_dir       = Path(image_dir)
    output_full_dir = Path(output_full_dir)
    output_crop_dir = Path(output_crop_dir)
    output_full_dir.mkdir(parents=True, exist_ok=True)
    output_crop_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco_check = json.load(f)
    valid_stems = {Path(img["file_name"]).stem for img in coco_check["images"]}

    # Load YOLO model via Ultralytics (supports .pt and .onnx)
    model_bbox = YOLO(yolo_model_path)

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    new_entries = []

    for idx, image_path in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] {image_path.name}")

        if image_path.stem not in valid_stems:
            print(f"  Label tidak ditemukan, skip: {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Failed to load: {image_path.name}")
            continue

        try:
            mask_final = generate_mask(image, segmenter, refiner)
        except Exception as e:
            print(f"  Mask generation failed: {e}")
            continue

        _, mask_bin = cv2.threshold(mask_final, 127, 255, cv2.THRESH_BINARY)
        if not np.any(mask_bin):
            print("  Empty mask, skipping")
            continue

        styles, _ = get_collar_styles(mask_bin, image)
        if not styles:
            print("  No styles generated, skipping")
            continue

        for style in styles:
            tag       = style["name"]
            aug_image = apply_shirt_style(image, style, mask_bin)

            full_name = f"{image_path.stem}_{tag}.jpg"
            cv2.imwrite(str(output_full_dir / full_name), aug_image)
            print(f"  Saved full : {full_name}")

            cropped, _ = run_yolo(model_bbox, aug_image, padding=padding, conf_threshold=conf_threshold)
            if cropped is not None:
                crop_name = f"{image_path.stem}_{tag}_crop.jpg"
                cv2.imwrite(str(output_crop_dir / crop_name), cropped)
                print(f"  Saved crop : {crop_name}")

                new_entries.append({
                    "orig_stem"   : image_path.stem,
                    "new_filename": crop_name,
                    "img_width"   : cropped.shape[1],
                    "img_height"  : cropped.shape[0],
                })

    print(f"\nSaving JSON ({len(new_entries)} entries)...")
    duplicate_annotations_json(ann_file, out_ann_file, new_entries)
    print("Augmentation complete.")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segmenter = MediaPipeSkinSegmenter("../../models/selfie_multiclass.tflite")

    refiner = SegRefinerWrapper(
        config_path     = "../../SegRefiner/configs/segrefiner/segrefiner_hr.py",
        checkpoint_path = "../../models/segrefiner_hr_latest.pth",
        device          = device,
        target_size     = 640,
    )

    run_augmentation_pipeline(
        image_dir       = str(DATASET_DIR / "data/all_images/full_images"),
        output_full_dir = str(DATASET_DIR / "data/all_images/full_images_augmentations/train"),
        segmenter       = segmenter,
        refiner         = refiner,
        ann_file        = str(DATASET_DIR / "data/annotations/train.json"),
        out_ann_file    = str(DATASET_DIR / "data/annotations/train_augmented.json"),
        output_crop_dir = str(DATASET_DIR / "data/images/train"),
        yolo_model_path = str(MODEL_BBOX_DIR / "good_augwed/weights/last.pt"),
        conf_threshold  = 0.25,
        padding         = 10,
    )