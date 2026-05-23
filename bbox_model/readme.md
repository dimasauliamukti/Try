# BBOX Model

Trains a YOLO model to detect and crop the neck region. The output model is used both in the main inference pipeline and as an assistant during keypoint annotation.

## Pipeline Overview

1. Annotate Images
2. Preprocess Data
3. Apply Augmentations
4. Train Model

---

## 1. Annotate Images

```bash
python src/GUI_BBOX.py
```

Two annotation modes are available:

### Manual Annotation

1. Click **Open Images** and select one or more images (`Shift + Click` or **Select All**).
2. Click and drag on the image to draw a bounding box around the neck region.
3. To reuse a box, click **Copy BBox** then **Paste to Next Image**.
4. Annotated bounding boxes are listed in the table at the bottom.
5. Labels are saved automatically on **Next**, or manually via the save icon:
   ```
   datasets/<your_folder>/labels/
   ```

### Semi-Manual Annotation (Model-Assisted)

1. Optionally click **Change BBox Model** to load a different `.pt` model for auto-detection.
2. Click **Open Images** and select images.
3. Click **Auto Detect** — boxes below the confidence threshold are flagged in the log panel.
4. Double-click any flagged entry in the log to jump to that image and correct it.
5. To reuse a box, click **Copy BBox** then **Paste to Next Image**.
6. Labels are saved automatically on **Next**, or manually via the save icon:
   ```
   datasets/<your_folder>/labels/
   ```
7. Click **Clear Log** to reset the low-confidence log.

> Only bounding boxes with a confidence score **≥ threshold** are saved. Boxes below the threshold are flagged but not saved.

---

## 2. Preprocess Data

1. Open `src/split.py` and set the input/output paths for images and labels.
2. The script splits data into **80% train / 20% val** by default.
3. Create a data.yaml on your <models/output_folder>, the data yaml will look like this:
```yaml
path: <your_dataset_path>
train: train/images
val: val/images

names:
  0: neck
```
3. Run:

```bash
python src/split.py
```

---

## 3. Apply Augmentations

1. Open `src/augmentations.py`.
2. Ensure model weights for MediaPipe and SegRefiner are set correctly.
3. The script blends both segmentation masks and generates synthetic neckline styles (turtleneck, V-neck, crewneck).
4. Set the output paths for augmented images and labels.
5. Run:

```bash
python src/augmentations.py
```

---

## 4. Train Model

1. Open `src/model_train.py` and set the desired output model name.
2. Run:

```bash
python src/model_train.py
```

The trained model is saved to `models/<output_model>/weights/`.

---

## Main Files

| File | Description |
|---|---|
| `src/GUI_BBOX.py` | Annotation GUI — manual and model-assisted bounding box labeling |
| `src/split.py` | Splits dataset into train and validation sets |
| `src/augmentations.py` | Augmented images with synthetic neckline styles |
| `src/moving_image.py` | Image file management and organization |
| `src/model_train.py` | YOLO BBOX model training |