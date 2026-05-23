from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

#DATASET FOLDER
DATASET_DIR = BASE_DIR /"datasets"
MODEL_DIR=BASE_DIR/"models"

SEGREFINER_DIR = BASE_DIR / "SegRefiner"
# if str(SEGREFINER_DIR) not in sys.path:
#     sys.path.insert(0, str(SEGREFINER_DIR))