from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR_KEYPOINTS = Path(__file__).resolve().parent.parent


#DATASET FOLDER
DATASET_DIR = BASE_DIR_KEYPOINTS /"datasets"

MODEL_BBOX_DIR=BASE_DIR/"bbox_model"/"models"
MODEL_MAIN_DIR=BASE_DIR/"models"
MODEL_KEYPOINTS_DIR=BASE_DIR_KEYPOINTS/"models"

SEGREFINER_DIR = BASE_DIR / "SegRefiner"
# if str(SEGREFINER_DIR) not in sys.path:
#     sys.path.insert(0, str(SEGREFINER_DIR))