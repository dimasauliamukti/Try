# Necklace Detection

A hybrid pipeline for detecting necklace anchor points on the neck area, combining deep learning (YOLO + HRNet), segmentation (MediaPipe + SegRefiner), and classical computer vision (OpenCV contours).

---

## Assets

| Asset | Destination | Link |
|---|---|---|
| BBOX Dataset | `bbox_model/datasets/` | [Download](https://drive.google.com/file/d/143kerk4WOerGZMqR-cAETb7rOcoQ0CQ_/view) |
| Keypoints Dataset | `keypoints_model/datasets/` | [Download](https://drive.google.com/file/d/143kerk4WOerGZMqR-cAETb7rOcoQ0CQ_/view) |
| BBOX Model | `bbox_model/models/` | [Download](https://drive.google.com/file/d/143kerk4WOerGZMqR-cAETb7rOcoQ0CQ_/view) |
| Keypoints Model | `keypoints_model/models/` | [Download](https://drive.google.com/file/d/143kerk4WOerGZMqR-cAETb7rOcoQ0CQ_/view) |
| SegRefiner Model | `models/` | [Download](https://drive.google.com/file/d/143kerk4WOerGZMqR-cAETb7rOcoQ0CQ_/view) |
| MediaPipe Model | `models/` | [Download](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite) |

---

## Project Structure

```
Necklace Detection/
├── assets/
├── src/                    # Inference code
├── models/                 # MediaPipe & SegRefiner models
├── SegRefiner/
├── Datasets/
├── bbox_model/             # BBOX training pipeline
│   ├── src/
│   ├── datasets/
│   ├── models/
│   ├── requirements.txt
│   └── README.md
├── keypoints_model/        # Keypoints training pipeline
│   ├── src/
│   │   ├── mmpose/
│   │   └── mmdeploy/
│   ├── datasets/
│   ├── models/
│   ├── requirements.txt
│   └── README.md
├── .github/
│   └── workflows/
│       └── deploy.yml
├── requirements.txt
└── README.md
```

---

## Getting Started
all of the requirements txt and this code was run on python 3.10 (Please use this version of python)
### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

**Inference:**
```bash
python src/inference.py
```
> Verify that model and image paths are correctly set in `src/config.py`.

**Local API:**
```bash
python src/api.py
```

---

## Inference Pipeline

### 1. Preprocessing
- Zoom into the face using Haar Cascade (fallback: 20% zoom if no face detected)
- Resize to `640×640` using letterbox

### 2. Segmentation
- Generate segmentation mask with MediaPipe
- Refine mask with SegRefiner
- Blend both masks to produce a neck and clothing region mask
- Resize mask back to original image dimensions

### 3. Necklace Region Detection
- Crop necklace region using BBOX model
- Fallback order: Haar Cascade crop → upper-region crop (`bbox.py`)

### 4. Keypoint Detection
- Detect keypoints with HRNet
- Map keypoints back to `640×640` space
- Snap keypoints to nearest mask contour on the same Y-axis
- Map keypoints back to original image dimensions

### 5. Output

```json
{
  "right_neck_shoulder_point": {
    "original_point": [523, 412],
    "resized_point": [318, 251],
    "score": 0.9821
  },
  "left_neck_shoulder_point": {
    "original_point": [441, 408],
    "resized_point": [267, 248],
    "score": 0.9754
  },
  "mask_area": [[120, 200], [121, 201], [122, 202]],
  "img_size": [1280, 720]
}
```

### 6. Failure Handling

| Condition | Message |
|---|---|
| Keypoint score < 35% | `Neck-shoulder area not clearly visible` |
| Hair covering shoulder > 40% | `Hair may be covering the shoulder area` |
| Image load failed | `Invalid Image` |
| Model load failed | `Model Load Error` |
| Detection failed | `Detection Failed` |

---

## Cloud Deployment (Modal)

### Prerequisites — Upload Models to Hugging Face

Create four HF repos named `bbox`, `refiner`, `keypoints`, and `person`, then upload each model:

| Model | Source Path | HF Repo |
|---|---|---|
| BBOX (YOLO) | `bbox_model/models/<folder>/weights/best.onnx` | `<username>/bbox` |
| SegRefiner | `models/segrefiner_hr_latest.pth` | `<username>/refiner` |
| Keypoints (HRNet) | `keypoints_model/models/<folder>/best.onnx` | `<username>/keypoints` |
| MediaPipe | `models/selfie_multiclass.tflite` | `<username>/person` |

After uploading, confirm repo IDs and filenames in `src/model_loader.py` match your HF repositories.

and then create your api key on huggingface, and set it in modal, and your local, place it in src/.env, example:
```env
HF_TOKEN=XXXXXXXXXXXXXXXXXXXXX
```
---

### Option 1 : Manual Deploy

```bash
modal token new
modal deploy modal_api.py
```

> Dependencies are installed inside Modal via `modal.Image` — no local installation needed.

---

### Option 2 : Auto Deploy via GitHub Actions (Recommended)

**Step 1:** Get your Modal token:
```bash
modal token new
```

**Step 2:** Add secrets to your GitHub repo under **Settings → Secrets and variables → Actions**:
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

**Step 3:** Push to `main` — the workflow in `.github/workflows/deploy.yml` handles the rest.

```
git push → GitHub Actions → Modal redeploy ✅
```

---

## Sub-Projects

- BBOX Model training → [`bbox_model/README.md`](./bbox_model/README.md)
- Keypoints Model training → [`keypoints_model/README.md`](./keypoints_model/README.md)

---

## Main Files

| File | Description |
|---|---|
| `src/inference.py` | Main inference pipeline |
| `src/api.py` | Local API server |
| `src/modal_api.py` | Modal deployment API |
| `src/config.py` | Global configuration |
| `src/bbox.py` | Necklace region detection and fallback logic |
| `src/keypoints.py` | HRNet keypoint inference |
| `src/preprocess.py` | Image preprocessing |
| `src/resize_image.py` | Letterbox resizing |
| `src/mediapipe_mp.py` | MediaPipe segmentation |
| `src/segmentation_mp.py` | SegRefiner mask refinement |
| `src/model_loader.py` | Model loader from HF Hub |
| `src/exception.py` | Error and failure handling |