from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
import io
import uvicorn
import cv2
import time
import json
import logging
import numpy as np
import logging
from contextlib import asynccontextmanager

import sys
import os

from exception import ModelLoadError,DetectionFailed
from preprocess import snap_to_nearest_contour,boundary_blend,mask_to_coords,to_json
from resize_image import resize,crop_upper_body,map_keypoint_to_original
from bbox import DetectNeck
from segmentation_mp import MediaPipeSkinSegmenter
from segrefiner import SegRefinerWrapper
from keypoints import DetectKeypoints
import torch 
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models() -> tuple:
    """
    Initialize and return all inference models required by the application.

    Loads the following models in order:
      1. MediaPipe skin segmenter for coarse mask generation.
      2. SegRefiner for boundary refinement of the segmentation mask.
      3. YOLO-based neck bounding-box detector.
      4. HRNet-based keypoint detector for neck/shoulder landmark localization.

    Returns:
        (segmenter, refiner, bbox_model, keypoints_model) where:
          - segmenter      : MediaPipeSkinSegmenter instance.
          - refiner        : SegRefinerWrapper instance.
          - bbox_model     : DetectNeck instance.
          - keypoints_model: DetectKeypoints instance.
    """
    segmenter = MediaPipeSkinSegmenter("../models/selfie_multiclass.tflite")
    #Segrefiner
    refiner = SegRefinerWrapper(
        config_path     = "../SegRefiner/configs/segrefiner/segrefiner_hr.py",
        checkpoint_path = "../models/segrefiner_hr_latest.pth",
        device          = device,
        target_size     = 640
    ) 
    #BBOX
    bbox_model=DetectNeck("../models/new_bbox3/weights/best.onnx")
    #Keypoints HRNET
    keypoints_model=DetectKeypoints("../keypoints_model/models/necklace_keypoints.onnx")
    
    return segmenter,refiner,bbox_model,keypoints_model


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.

    On startup, loads all ML models into the global ``models`` dict so they
    are shared across requests without reloading. On shutdown the context
    exits cleanly (models are released when the process ends).

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application after models are loaded.
    """
    models["mediapipe"], models["segrefiner"], models["bbox"], models["keypoints"] = load_models()
    yield
    
    
    
app = FastAPI(lifespan=lifespan)

    
@app.exception_handler(ModelLoadError)
async def model_load_handler(request, exc):
    """
    Handle ModelLoadError exceptions raised during inference.

    Logs the error and returns a JSON response with HTTP status code
    defined by the exception, along with an ``MODEL_LOAD_ERROR`` error code.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc:     The ModelLoadError instance containing the error message
                 and status code.

    Returns:
        JSONResponse with error_code ``MODEL_LOAD_ERROR`` and the exception message.
    """
    logger.error(str(exc))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": "MODEL_LOAD_ERROR", "message": str(exc)}
    )

@app.exception_handler(DetectionFailed)
async def detection_failed_handler(request, exc):
    """
    Handle DetectionFailed exceptions raised when a model cannot produce results.

    Logs the error and returns a JSON response with HTTP status code
    defined by the exception, along with a ``DETECTION_FAILED`` error code.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc:     The DetectionFailed instance containing the error message
                 and status code.

    Returns:
        JSONResponse with error_code ``DETECTION_FAILED`` and the exception message.
    """
    logger.error(str(exc))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": "DETECTION_FAILED", "message": str(exc)}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """
    Handle ValueError exceptions, typically raised for invalid or unreadable images.

    Logs the error and returns a 400 JSON response with an ``INVALID_IMAGE``
    error code, signalling a bad client request.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc:     The ValueError instance containing the error message.

    Returns:
        JSONResponse (HTTP 400) with error_code ``INVALID_IMAGE`` and the exception message.
    """
    logger.error(f"Value error: {exc})")
    return JSONResponse(
        status_code=400,
        content={"error_code": "INVALID_IMAGE", "message": str(exc)}
    )

@app.exception_handler(Exception)
async def unexpected_handler(request, exc):
    """
    Catch-all handler for any unhandled exceptions.

    Logs the unexpected error and returns a 500 JSON response with an
    ``INTERNAL_ERROR`` error code to avoid exposing raw tracebacks to clients.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc:     The unhandled Exception instance.

    Returns:
        JSONResponse (HTTP 500) with error_code ``INTERNAL_ERROR`` and the exception message.
    """
    logger.error(f"Unexpected: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error_code": "INTERNAL_ERROR", "message": str(exc)}
    )


@app.get("/")
async def root():
    """
    Root health-check endpoint.

    Returns:
        dict: A simple status message confirming the server is running.
    """
    return {"message": "Server is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze an uploaded portrait image and return neck/shoulder keypoints and mask data.

    Full inference pipeline:
      1. Decode uploaded image bytes to a BGR NumPy array.
      2. Crop the upper-body region and resize to the model's expected input size.
      3. Run MediaPipe segmentation to obtain coarse skin/neck masks.
      4. Refine mask boundaries with SegRefiner.
      5. Blend MediaPipe and SegRefiner masks for the final mask.
      6. Extract the neck-body mask coordinates.
      7. Detect the neck bounding box with the YOLO model.
      8. Detect neck/shoulder keypoints with HRNet inside the bounding box.
      9. Snap keypoints to the nearest mask contour for sub-pixel accuracy.
      10. Map refined keypoints back to the original image coordinate space.

    Args:
        file: Uploaded image file (multipart/form-data). Must be a valid
              image decodable by OpenCV (JPEG, PNG, etc.).

    Raises:
        ValueError: If the uploaded file cannot be decoded as an image.

    Returns:
        dict: JSON object containing:
          - right_neck_shoulder_point (dict):
              - original_point (list[int]): [x, y] in the original image space.
              - resized_point  (list[int]): [x, y] in the resized image space.
              - score          (float)    : Detection confidence score.
          - left_neck_shoulder_point (dict):
              - original_point (list[int]): [x, y] in the original image space.
              - resized_point  (list[int]): [x, y] in the resized image space.
              - score          (float)    : Detection confidence score.
          - mask_area (list[list[int]]): Contour coordinates of the neck-body mask.
          - img_size  (list[int])      : [width, height] of the original image.
    """
    logger.info(f"Analyze request: {file.filename}")
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    

    if img is None:
        raise ValueError()
    
    h,w=img.shape[:2]
    face, face_info = crop_upper_body(img)
    image, r, top, left = resize(face)
    mask, image_seg, skin_mask,neck_face,neck_body = models["mediapipe"].segment(image)
    # Segrefiner
    mask_segrefiner = models["segrefiner"].run(image, mask, iterations=1)
    mask_final  = boundary_blend(mask, mask_segrefiner, boundary_width=3) 
    neckface_final =boundary_blend(neck_face, mask_segrefiner, boundary_width=3) 
    neckbody_final  = boundary_blend(neck_body, mask_segrefiner, boundary_width=3)
    mask_coordinate=mask_to_coords(neckbody_final)
    bbox,bbox_coords,_=models["bbox"].detect_bbox(image)
    bx1, by1, bx2, by2= bbox_coords

    #Detect KP
    keypoints=models["keypoints"].detect_kp(bbox)
    keypoints_on_resized = [(int(x) + bx1, int(y) + by1, float(score)) for x, y, score in keypoints]
    keypoints_snapped = snap_to_nearest_contour(neckface_final,mask_final, keypoints_on_resized)

    rx,ry,rscore,rval=keypoints_snapped[0]
    lx,ly,lscore,lval=keypoints_snapped[1]
    
    if not rval or not lval:
        logger.warning("Snap Failed, Using Raw Keypoints")
        
    keypoints_original =[map_keypoint_to_original(x, y, r, top, left, face_info) for x, y, _, _ in keypoints_snapped]
    
    output_json={
        "right_neck_shoulder_point":{
            "original_point":to_json(list(keypoints_original[0])),
            "resized_point":to_json([rx,ry]),
            "score": to_json(rscore)
            },
        "left_neck_shoulder_point":{
            "original_point":to_json(list(keypoints_original[1])),
            "resized_point":to_json([lx,ly]),
            "score": to_json(lscore)    
            },
        "mask_area":to_json(mask_coordinate),
        "img_size":to_json([w,h])   
    }
    logger.info(f"Analyze success: {file.filename}")
    return output_json 


@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    """
    Run the full inference pipeline on an uploaded image and return a JPEG with
    detected keypoints drawn on it, for visual debugging and inspection.

    Executes the same pipeline as ``/analyze`` (segmentation → bounding box →
    keypoint detection → contour snapping) and then renders the two snapped
    neck/shoulder keypoints as green circles directly on the resized image.

    Args:
        file: Uploaded image file (multipart/form-data). Must be a valid
              image decodable by OpenCV (JPEG, PNG, etc.).

    Raises:
        ValueError: If the uploaded file cannot be decoded as an image.

    Returns:
        StreamingResponse: A JPEG image (media_type ``image/jpeg``) with the
        two snapped keypoints overlaid as filled green circles (radius 2 px).
    """
    logger.info(f"Visualize request: {file.filename}")

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError()
    
    face, face_info = crop_upper_body(img)
    image, r, top, left = resize(face)
    mask, image_seg, skin_mask,neck_face,neck_body = models["mediapipe"].segment(image)
    # Segrefiner
    mask_segrefiner = models["segrefiner"].run(image, mask, iterations=1)
    mask_final  = boundary_blend(mask, mask_segrefiner, boundary_width=3) 
    neckface_final =boundary_blend(neck_face, mask_segrefiner, boundary_width=3) 
    neckbody_final  = boundary_blend(neck_body, mask_segrefiner, boundary_width=3)
    mask_coordinate=mask_to_coords(neckbody_final)
    bbox,bbox_coords,_=models["bbox"].detect_bbox(image)
    bx1, by1, bx2, by2= bbox_coords

    #Detect KP
    keypoints=models["keypoints"].detect_kp(bbox)
    keypoints_on_resized = [(int(x) + bx1, int(y) + by1, float(score)) for x, y, score in keypoints]
    keypoints_snapped = snap_to_nearest_contour(neckface_final,mask_final, keypoints_on_resized)

    rx,ry,rscore,rval=keypoints_snapped[0]
    lx,ly,lscore,lval=keypoints_snapped[1]
    
    if not rval or not lval:
        logger.warning("Snap Failed, Using Raw Keypoints")
    
    cv2.circle(image, (rx,ry), 2, (0,255,0),-1)
    cv2.circle(image, (lx,ly), 2, (0,255,0),-1)
    
    _, buffer = cv2.imencode(".jpg", image)
    logger.info(f"Visualize success: {file.filename}")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.get("/health")
def health():
    """
    Lightweight health-check endpoint for load balancers and monitoring tools.

    Returns:
        dict: ``{"status": "ok"}`` when the server is running normally.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)