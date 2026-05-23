import json
import os
import cv2
from config_keypoints import DATASET_DIR


def yolo_to_coco_json(image_dir, label_dir, output_json, keypoint_names):
    """
    Convert YOLO keypoint annotations to COCO JSON format.

    Reads all .txt label files from label_dir, matches each to its corresponding
    image in image_dir, denormalizes bounding boxes and keypoints using the image
    dimensions, and writes the result as a COCO-format JSON file.

    Args:
        image_dir (str):        Directory containing source images (.jpg, .png, .jpeg).
        label_dir (str):        Directory containing YOLO .txt annotation files.
        output_json (str):      Output path for the resulting COCO .json file.
        keypoint_names (list):  Ordered list of keypoint names matching YOLO annotation order.
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": keypoint_names,
            "skeleton": []
        }]
    }

    ann_id = 1
    image_id = 1

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        file_base = os.path.splitext(label_file)[0]

        # Find the matching image file (supports .jpg, .png, .jpeg)
        img_name = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(image_dir, file_base + ext)
            if os.path.exists(candidate):
                img_name = file_base + ext
                break

        if img_name is None:
            print(f"[WARNING] No matching image found for label: {label_file} — skipping.")
            continue

        # Read image dimensions for denormalization
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path} — skipping.")
            continue
        h, w, _ = img.shape

        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue

                # YOLO format: class cx cy w h [kx ky kv ...]
                bbox_width  = parts[3] * w
                bbox_height = parts[4] * h
                bbox_x      = (parts[1] * w) - (bbox_width / 2)
                bbox_y      = (parts[2] * h) - (bbox_height / 2)

                # Extract and denormalize keypoints
                keypoints_raw = parts[5:]
                num_keypoints = len(keypoints_raw) // 3
                coco_kpts = []

                for i in range(num_keypoints):
                    kx = keypoints_raw[i * 3]     * w
                    ky = keypoints_raw[i * 3 + 1] * h
                    kv = int(keypoints_raw[i * 3 + 2])  # 0: hidden, 1: occluded, 2: visible
                    coco_kpts.extend([kx, ky, kv])

                coco_format["annotations"].append({
                    "id":            ann_id,
                    "image_id":      image_id,
                    "category_id":   1,
                    "segmentation":  [],
                    "area":          bbox_width * bbox_height,
                    "bbox":          [bbox_x, bbox_y, bbox_width, bbox_height],
                    "keypoints":     coco_kpts,
                    "num_keypoints": num_keypoints,
                    "iscrowd":       0
                })
                ann_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"[INFO] Saved {ann_id - 1} annotations across {image_id - 1} images → {output_json}")


# ── Configuration ─────────────────────────────────────────────────────────────

KEYPOINT_NAMES = ["neck_right", "neck_left"]  # Must match YOLO annotation order

SPLITS = {
    # "val": {
    #     "image_dir":   DATASET_DIR / "data" / "images" / "val",
    #     "label_dir":   DATASET_DIR / "data" / "val"    / "labels",
    #     "output_json": DATASET_DIR / "data" / "annotations" / "val.json",
    # },
    # Uncomment to also convert the training split:
    "train": {
        "image_dir":   DATASET_DIR / "data" / "images"/ "train" ,
        "label_dir":   DATASET_DIR / "data" / "train" / "labels",
        "output_json": DATASET_DIR / "data" / "annotations" / "train.json",
    },
}

if __name__ == "__main__":
    for split_name, paths in SPLITS.items():
        print(f"\n[INFO] Converting split: {split_name}")
        yolo_to_coco_json(
            image_dir   = str(paths["image_dir"]),
            label_dir   = str(paths["label_dir"]),
            output_json = str(paths["output_json"]),
            keypoint_names = KEYPOINT_NAMES,
        )