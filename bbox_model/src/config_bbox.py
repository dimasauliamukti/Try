from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR_BBOX = Path(__file__).resolve().parent.parent


#DATASET FOLDER
DATASET_DIR = BASE_DIR_BBOX /"datasets"
MODEL_DIR=BASE_DIR_BBOX/"models"

SEGREFINER_DIR = BASE_DIR / "SegRefiner"
# if str(SEGREFINER_DIR) not in sys.path:
#     sys.path.insert(0, str(SEGREFINER_DIR))