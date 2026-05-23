import os
import random
import shutil
from config_bbox import DATASET_DIR

# Path
image_all_dir = DATASET_DIR / "data" / "all_images" # Source directory containing all Images files (.jpg, .jpeg, .png)
label_all_dir = DATASET_DIR / "data" / "all_labels" # Source directory containing all raw YOLO label files (.txt)
train_label_dir = DATASET_DIR / "data"            / "train" / "labels" # Destination for training split labels
val_label_dir   = DATASET_DIR / "data" / "val" / "labels" # Destination for val split labels
train_image_dir = DATASET_DIR / "data"            / "train" / "images" # Destination for training split images
val_image_dir   = DATASET_DIR / "data" / "val" / "images" # Destination for val split images

# Config
split_ratio  = 0.8
random_seed  = 42
image_extensions = [".jpg", ".png", ".jpeg"]

for d in [train_label_dir, val_label_dir, train_image_dir, val_image_dir]:
    os.makedirs(d, exist_ok=True)


def split_and_copy(
    label_src_dir, image_src_dir,
    train_label_dst, val_label_dst,
    train_image_dst, val_image_dst,
    split_ratio=split_ratio, seed=random_seed,
):
    """
    Split YOLO label files into train/val sets and copy matching images.

    Reads all .txt files from label_src_dir, splits them by split_ratio,
    then copies labels and their matching images to the respective destinations.
    Images are matched by base name with extensions .jpg, .png, or .jpeg.
    Missing images are reported per split but do not raise an error.

    Args:
        label_src_dir:   Directory containing all raw YOLO .txt label files.
        image_src_dir:   Directory containing all source images.
        train_label_dst: Destination directory for training labels.
        val_label_dst:   Destination directory for validation labels.
        train_image_dst: Destination directory for training images.
        val_image_dst:   Destination directory for validation images.
        split_ratio:     Fraction used for training, rest goes to val (default 0.8).
        seed:            Random seed for reproducibility (default 42).
    """
    random.seed(seed)

    files = [f for f in os.listdir(label_src_dir) if f.endswith(".txt")]
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    splits = {
        "train": (files[:split_index], train_label_dst, train_image_dst),
        "val":   (files[split_index:], val_label_dst,   val_image_dst),
    }

    for split_name, (split_files, label_dst, image_dst) in splits.items():
        missing = []
        for fname in split_files:
            # Copy label
            shutil.copy(os.path.join(label_src_dir, fname), os.path.join(label_dst, fname))

            # Copy matching image
            base = os.path.splitext(fname)[0]
            for ext in image_extensions:
                img_src = os.path.join(image_src_dir, base + ext)
                if os.path.exists(img_src):
                    shutil.copy(img_src, os.path.join(image_dst, base + ext))
                    break
            else:
                missing.append(base)

        print(f"[{split_name:>5}] labels: {len(split_files)}"
              + (f" | image not found: {missing}" if missing else ""))

    print(f"\nTotal : {len(files)}")
    print(f"Train : {split_index}")
    print(f"Val   : {len(files) - split_index}")


split_and_copy(
    label_src_dir   = label_all_dir,
    image_src_dir   = image_all_dir,
    train_label_dst = train_label_dir,
    val_label_dst   = val_label_dir,
    train_image_dst = train_image_dir,
    val_image_dst   = val_image_dir,
)