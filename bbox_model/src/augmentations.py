"""
augmentation_pipeline_yolo.py
Augments a dataset of person images by overlaying four collar/shirt styles
(turtleneck, v-neck, crewneck, collar shirt) onto the detected torso region.

Output format: YOLO TXT label files only.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
import sys
import os
import torch
from config_bbox import BASE_DIR, DATASET_DIR, MODEL_DIR, SEGREFINER_DIR

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from segmentation_mp import MediaPipeSkinSegmenter
from segrefiner import SegRefinerWrapper


STYLE_COLOR_MAP = {
    "turtleneck"  : (30,  30,  30),
    "vneck"       : (0,   0,  200),
    "crewneck"    : (0,  128,   0),
    "collar_shirt": (42,  82, 139),
}


# Mask utilities

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
    mask, _, _, _, skin_mask = segmenter.segment(image_rgb)

    mask_segrefiner = refiner.run(image, mask, iterations=1)
    return boundary_blend(skin_mask, mask_segrefiner, boundary_width=3)


# Collar-style geometry
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


# Collar-shirt triangle helpers

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
    """Compute the three vertices of the left collar-flap triangle."""
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
    """Compute the three vertices of the right collar-flap triangle."""
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


# Image compositing

def blend_color_to_mask(image: np.ndarray, side_mask: np.ndarray, color: tuple) -> np.ndarray:
    """
    Composite a solid color (with slight noise) onto the image wherever
    side_mask is non-zero, using a Gaussian-blurred soft edge.
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
      - "turtleneck"   
      - "vneck"        
      - "crewneck"     
      - "collar_shirt" 
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


# Annotation helpers — YOLO TXT
def duplicate_annotations_yolo(
    label_dir: str,
    out_label_dir: str,
    new_entries: list[dict],
) -> None:
    """
    Copy YOLO TXT label files for each augmented image.

    For every entry in new_entries, the label file of the original image
    is copied and renamed to match the augmented image filename.

    Args:
        label_dir:     Directory containing original YOLO TXT label files.
        out_label_dir: Directory where new label files will be written.
        new_entries:   List of dicts with keys:
                         - orig_stem    (str): stem of the original image.
                         - new_filename (str): filename of the augmented image.
    """
    label_dir     = Path(label_dir)
    out_label_dir = Path(out_label_dir)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    success_count = skip_count = 0
    for entry in new_entries:
        src_txt = label_dir / f"{entry['orig_stem']}.txt"
        dst_txt = out_label_dir / f"{Path(entry['new_filename']).stem}.txt"

        if not src_txt.exists():
            skip_count += 1
            continue

        shutil.copy2(src_txt, dst_txt)
        success_count += 1

    print(f"YOLO TXT saved    : {out_label_dir}")
    print(f"Succeeded         : {success_count}")
    print(f"Skipped           : {skip_count}")


# Main pipeline

def run_augmentation_pipeline(
    image_dir: str,
    output_full_dir: str,
    segmenter,
    refiner,
    label_dir: str,
    out_label_dir: str,
) -> None:
    """
    Run the full shirt-style augmentation pipeline on a directory of images.

    For each input image, four augmented variants (turtleneck, v-neck, crewneck,
    collar shirt) are generated and saved. YOLO TXT label files are duplicated
    to match the augmented image filenames.

    Args:
        image_dir:       Directory of input images (.jpg / .png).
        output_full_dir: Directory to write full-size augmented images.
        segmenter:       MediaPipeSkinSegmenter instance.
        refiner:         SegRefinerWrapper instance.
        label_dir:       Directory of original YOLO TXT label files.
        out_label_dir:   Directory to write augmented YOLO TXT labels.
    """
    image_dir       = Path(image_dir)
    output_full_dir = Path(output_full_dir)
    output_full_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    new_entries = []

    for idx, image_path in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path.name}")
            continue

        try:
            mask_final = generate_mask(image, segmenter, refiner)
        except Exception as e:
            print(f"Mask generation failed: {e}")
            continue

        _, mask_bin = cv2.threshold(mask_final, 127, 255, cv2.THRESH_BINARY)
        if not np.any(mask_bin):
            print("Empty mask, skipping")
            continue

        styles, _ = get_collar_styles(mask_bin, image)
        if not styles:
            print("No styles generated, skipping")
            continue

        for style in styles:
            tag       = style["name"]
            aug_image = apply_shirt_style(image, style, mask_bin)

            full_name = f"{image_path.stem}_{tag}.jpg"
            cv2.imwrite(str(output_full_dir / full_name), aug_image)
            print(f"Saved: {full_name}")

            new_entries.append({
                "orig_stem"   : image_path.stem,
                "new_filename": full_name,
            })

    print(f"Saving YOLO TXT ({len(new_entries)} entries)...")
    duplicate_annotations_yolo(label_dir, out_label_dir, new_entries)
    print("Augmentation complete.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = MediaPipeSkinSegmenter("../models/selfie_multiclass.tflite")
    # SegRefiner — high-resolution boundary refinement
    refiner = SegRefinerWrapper(
        config_path     = "../SegRefiner/configs/segrefiner/segrefiner_hr.py",
        checkpoint_path = "../model/segrefiner_hr_latest.pth",
        device          = device,
        target_size     = 640
    )

    run_augmentation_pipeline(
        image_dir       = str(DATASET_DIR / "data/train/images"),
        output_full_dir = str(DATASET_DIR / "data/train/images"),
        segmenter       = segmenter,
        refiner         = refiner,
        label_dir       = str(DATASET_DIR / "data/train/labels"),
        out_label_dir   = str(DATASET_DIR / "data/train/labels"),
    )