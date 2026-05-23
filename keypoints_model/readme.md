# Keypoints Model

Trains an HRNet model to detect necklace anchor points (left and right shoulder) within the cropped neck region. The output model is used in the main inference pipeline.

## Pipeline Overview

1. Annotate Images
2. Preprocess Data
3. Apply Augmentations
4. Create a venv/change a venv
5. Train Model
6. Convert Model to ONNX

---

## 1. Annotate Images

```bash
python src/gui.py
```

### Setup

1. Optionally click **Change BBox Model** to load a different `.onnx` model for neck detection.
2. Click **Select Images** and select one or more images (`Shift + Click` or **Select All**).
3. Click **Set Crop Folder** and point to your output directory for cropped images:
   ```
   datasets/<your_folder>/crop_images/
   ```
4. Click **Set YOLO Folder** and point to your output directory for YOLO keypoint labels:
   ```
   datasets/<your_folder>/labels/
   ```

### Segment & Crop

5. Click **Segment + Crop Neck** to auto-detect and crop the neck region.

> If the neck cannot be detected, a warning will appear — skip that image and move on.

### Keypoint Annotation

6. Place two keypoints on the cropped neck image:

   | Keypoint | Label | Position |
   |---|---|---|
   | **KP1** | Keypoint 0 — Right | Left side of the image (person's right shoulder) |
   | **KP2** | Keypoint 1 — Left | Right side of the image (person's left shoulder) |

   > Click as close as possible to the skin or shoulder line for accurate placement.

7. Use **Reset KP1 (Right)** or **Reset KP2 (Left)** to redo a misplaced keypoint.
8. **Snap to Contour** is ON by default — keypoints snap to the nearest contour. Toggle it OFF for free placement.

### View Controls

| Toggle | Function |
|---|---|
| **Overlay** | Show / hide segmentation overlay |
| **Keypoints** | Show / hide placed keypoints |
| **Context View** | Show the original (un-cropped) image for reference |

### Saving & Navigation

9. Labels and cropped images are saved automatically on **Next**. To save manually, click **Save Crop Image** or **Export YOLO**. The Full Images is be saven to you can check it in: 
```
datasets/<your_folder>/crop_images/full_images
```
> Metadata (keypoint coordinates relative to the crop) is also saved. If you later re-crop with a different BBOX model, keypoints will be automatically remapped to the new crop.

### Load Previous Annotations

10. Click **Load Annotation + Map JSON**, then navigate to the folder containing previously saved labels. Annotations are matched by filename.

---

## 2. Preprocess Data

1. Open `src/split_dataset.py` and configure the following paths:

   | Variable | Description |
   |---|---|
   | `image_all_dir` | All source images |
   | `label_all_dir` | All source YOLO `.txt` labels |
   | `train_image_dir` | Training image output |
   | `val_image_dir` | Validation image output |
   | `train_label_dir` | Training label output |
   | `val_label_dir` | Validation label output |

2. Optionally adjust split ratio and seed:
   ```python
   split_ratio = 0.8  # 80% train, 20% val
   random_seed = 42
   ```

3. Run:
   ```bash
   python src/split_dataset.py
   ```

   > Images with no matching label are listed in the output and skipped without error.

### Convert YOLO to COCO JSON

4. Open `src/convert2json.py` and configure:

   | Variable | Description |
   |---|---|
   | `image_dir` | Source images for this split |
   | `label_dir` | YOLO `.txt` labels |
   | `output_json` | Output path for the COCO `.json` file |
   | `KEYPOINT_NAMES` | Ordered keypoint names, e.g. `["neck_right", "neck_left"]` |

   > To also convert the training split, uncomment the `"train"` block inside `SPLITS`.
5. Run :
```
python conver2json.py
```
6. Expected dataset structure after preprocessing:
   ```
   datasets/<your_dataset_folder>/
   ├── annotations/
   │   ├── train.json
   │   └── val.json
   ├── images/
   │   ├── train/
   │   └── val/
   ├── train/
   │   └── labels/
   └── val/
       └── labels/
   ```

---

## 3. Apply Augmentations

1. Open `src/augmentations.py`.
2. Ensure model weights for MediaPipe and SegRefiner are set correctly.
3. The script blends both segmentation masks and generates synthetic neckline styles (turtleneck, V-neck, crewneck).
4. Set the output paths for augmented images and labels. Configure:
    run_augmentation_pipeline(
        image_dir       = str(DATASET_DIR / "data/old_images_train"),
        output_full_dir = str(DATASET_DIR / "data/img/train_augmented_full"),
        segmenter       = segmenter,
        refiner         = refiner,
        ann_file        = str(DATASET_DIR / "data/annotations/train.json"),
        out_ann_file    = str(DATASET_DIR / "data/annotations/train_augmented.json"),
        output_crop_dir = str(DATASET_DIR / "data/images/train"),
        yolo_model_path = str(MODEL_BBOX_DIR / "good_augwed/weights/last.onnx"),
    )
   | Variable | Description |
   |---|---|
   | `image_dir` | Source images for this split |
   | `label_dir` | YOLO `.txt` labels |
   | `output_json` | Output path for the COCO `.json` file |
   | `KEYPOINT_NAMES` | Ordered keypoint names, e.g. `["neck_right", "neck_left"]` |

   > To also convert the training split, uncomment the `"train"` block inside `SPLITS`.
5. Run:

```bash
python src/augmentations.py
```

6. Rename the output from `train_augmented.json` to `train.json` to use augmented data for training.

---
## 4. Installation
> Train and Convert need its own isolated environment due to strict version dependencies (mmcv, torch). Do not install into an existing environment shared with other.

1. Create and activate a new virtual environment:

   ```bash
   python -m venv venv_keypoints
   source venv_keypoints/bin/activate        # Linux / macOS
   venv_keypoints\Scripts\activate           # Windows
   ```

2. Install dependencies:

   ```bash
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
   pip install -r requirements.txt
   ```

## 5. Train Model

### Dataset Config

1. Open `src/dataset_config.py`. This file defines the MMPose dataset configuration:
   - Keypoints: `neck_right` (id: 0), `neck_left` (id: 1)
   - Skeleton: connecting line between both keypoints
   - Joint weights: `1.0` for both keypoints
   - Sigmas: `0.025` for high-precision OKS evaluation

2. Verify `keypoint_info` and `skeleton_info` match your dataset before training.

### Training Config

1. Open `model_train.py` and set the following:

   ```python
   dataset_path = '../datasets/<your_dataset_folder>'
   work_dir = '../models/hrnet_necklace'
   ```

2. Tunable hyperparameters:

   | Parameter | Default | Description |
   |---|---|---|
   | `input_size` | `(288, 128)` | Input resolution — adjust to match H:W ratio of your crops |
   | `heatmap_size` | `(72, 32)` | Must be exactly 1/4 of `input_size` |
   | `max_epochs` | `100` | Total training epochs |
   | `lr` | `0.0005` | Learning rate |
   | `weight_decay` | `0.01` | Optimizer weight decay |
   | `milestones` | `[60, 90]` | Epochs where LR drops by 0.1× |
   | `batch_size` | `8` | Images per batch |
   | `sigma` | `1.5` | Heatmap gaussian size — larger = softer heatmap |
   | `rotate_factor` | `45` | Max rotation augmentation (degrees) |
   | `scale_factor` | `[0.6, 1.4]` | Random scale range for augmentation |

3. Checkpoint settings in `default_hooks`:
   ```python
   checkpoint=dict(
       interval=10,           # save every 10 epochs
       save_best='coco/AP',   # save best checkpoint by AP score
       max_keep_ckpts=5       # keep only the 5 most recent checkpoints
   )
   ```

### Run Training

```bash
python mmpose/tools/train.py model_train.py
```

The best checkpoint is saved as `best_coco_AP_epoch_XX.pth` in `models/<your_output_folder>/`.

---

## 6. Convert Model to ONNX

### Conversion Config

1. Open `configs/necklace/deploy_necklace.py` and set:

   | Parameter | Default | Description |
   |---|---|---|
   | `save_file` | `best.onnx` | Output ONNX filename |
   | `input_shape` | `(288, 128)` | Must match `input_size` from training |
   | `opset_version` | `11` | ONNX opset version |

2. Go to `models/<your_model_folder>/model_train.py` search for `metainfo` and changes:
```bash
#Before
metainfo=dict(from_file='config_onnx.py'),

#After
metainfo=dict(from_file='../src/config_onnx.py'),
```
### Run Conversion

```bash
python mmdeploy/tools/torch2onnx.py \
config_onnx.py \
../models/<your_model_folder>/model_train.py \
../models/<your_model_folder>/best_coco_AP_epoch_30.pth \
../datasets/<crop_images>.jpg \
--work-dir ../models/best \
--device cpu
```

The converted model is saved to `models/<your_model_folder>/best.onnx`.

---

## Main Files

| File | Description |
|---|---|
| `src/gui.py` | Annotation GUI — keypoint labeling with automatic neck detection and cropping |
| `src/split_dataset.py` | Splits dataset into train and validation sets |
| `src/convertdataset.py` | Converts YOLO `.txt` keypoint labels to COCO `.json` format |
| `src/augmentations.py` | Augmented images with synthetic neckline styles |
| `src/moving_image.py` | Image file management and organization |
| `src/model_train.py` | HRNet keypoint model training |
| `src/custom_necklace.py` | MMPose dataset configuration |
| `src/hrnet_w32_necklace.py` | HRNet-W32 training configuration |
| `src/deploy_necklace.py` | MMDeploy ONNX conversion configuration |