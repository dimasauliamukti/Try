from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv
# model_segmentation,model_refiner,model_bbox,model_keypoints
load_dotenv()
def model_segmentation():
    """Load Segmentation Model
    """
    model_path = hf_hub_download(
        repo_id="daulqqqq/person",
        filename="selfie_multiclass.tflite",
        token=os.getenv("HF_TOKEN")
    )
    return model_path

def model_refiner():
    """Load Necklace Model
    """
    model_path = hf_hub_download(
        repo_id="daulqqqq/refiner",
        filename="segrefiner_hr_latest.pth",
        token=os.getenv("HF_TOKEN")
    )
    return model_path

def model_bbox():
    """Load Necklace Model
    """
    model_path = hf_hub_download(
        repo_id="daulqqqq/bbox",
        filename="best.onnx",
        token=os.getenv("HF_TOKEN")
    )
    return model_path

def model_keypoints():
    """Load Necklace Model
    """
    model_path = hf_hub_download(
        repo_id="daulqqqq/keypoints",
        filename="end2end.onnx",
        token=os.getenv("HF_TOKEN")
    )
    return model_path