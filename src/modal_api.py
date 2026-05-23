import modal
import time


# Requirements
REQUIREMENTS = [
    "fastapi==0.111.0",
    "python-dotenv==1.2.1",
    "uvicorn==0.30.6",
    "requests>=2.31.0",
    "matplotlib==3.9.4",
    "numpy==1.26.4",
    "pillow==10.4.0",
    "scipy==1.13.1",
    "six==1.17.0",
    "tqdm==4.67.3",
    "terminaltables==3.1.10",
    "pycocotools==2.0.8",
    "opencv-python-headless==4.10.0.84",
    "opencv-contrib-python-headless==4.10.0.84",
    "mediapipe==0.10.18",
    "huggingface-hub>=0.20.0",
]

app = modal.App("neckdetection")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04",
        add_python="3.10"
    )
    .env({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "DEBIAN_FRONTEND": "noninteractive",
        "GLOG_minloglevel": "3",           # ← tambahkan
        "TF_CPP_MIN_LOG_LEVEL": "3",       # ← tambahkan
        "MEDIAPIPE_DISABLE_GPU": "1",      # ← tambahkan
        "ABSL_MIN_LOG_LEVEL": "3",         # ← tambahkan
    })
    .apt_install(
        "gcc", "g++", "build-essential",
        "libgl1", "libglib2.0-0",
        "libsm6", "libxext6",
        "libxrender-dev", "libgomp1",
        "curl", "git"
    )
    .pip_install("setuptools", "wheel", "pip")
    .pip_install("numpy==1.26.4")
    .pip_install(
        "torch==1.13.1+cu117",
        "torchvision==0.14.1+cu117",
        extra_index_url="https://download.pytorch.org/whl/cu117"
    )
    .pip_install(
        "mmcv-full==1.7.1",
        find_links="https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html"
    )
    .pip_install(*REQUIREMENTS)
    .pip_install("numpy==1.26.4")    
    .pip_install("onnxruntime-gpu==1.13.1")
    .add_local_dir(".", remote_path="/root/src")
    .add_local_dir(r"./SegRefiner", remote_path="/root/SegRefiner")
)

volume = modal.Volume.from_name("model-weights", create_if_missing=True)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
class DetectNecklacePoint:
    """
    Modal class that serves the neck/shoulder detection pipeline as a
    GPU-accelerated ASGI application on Modal's serverless infrastructure.

    Models are loaded once at container startup via ``load_models`` and shared
    across all requests. The ASGI endpoint is exposed through ``endpoint``,
    which mounts a FastAPI app with ``/analyze``, ``/visualize``, and
    ``/health`` routes.

    Raises:
        ModelLoadError: If any model fails to load during ``load_models``.
        DetectionFailed: If inference fails during ``analyze`` or ``visualize``.
        ShoulderNeckNotVisible: If the shoulder/neck region cannot be detected.
    """

    @property
    def logger(self):
        """
        Build and return a root logger configured to write to stdout.

        Reconfigures ``basicConfig`` on every access so the logger is always
        available regardless of Modal's container initialisation order.

        Returns:
            logging.Logger: Root logger instance with INFO level and timestamp format.
        """
        import logging
        import sys
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True
        )
        return logging.getLogger()

    @modal.enter()
    def load_models(self):
        """
        Load all ML models into instance attributes at container startup.

        Called once by Modal when the container starts (``@modal.enter``).
        Initialises MediaPipe, SegRefiner, YOLO bbox, and HRNet keypoint models,
        then performs a GPU warm-up pass with a dummy image to avoid cold-start
        latency on the first real request.

        Models loaded:
          - ``self.mediapipe``  : MediaPipeSkinSegmenter for coarse mask generation.
          - ``self.segrefiner`` : SegRefinerWrapper for boundary refinement.
          - ``self.bbox``       : DetectNeck for neck bounding-box detection.
          - ``self.keypoints``  : DetectKeypoints for neck/shoulder landmark detection.

        Raises:
            ModelLoadError: If any model fails to initialise.
        """
        import os
        import sys

        os.environ["GLOG_minloglevel"] = "3"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
        os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
        import torch
        import numpy as np
        import warnings
        
        import logging

        warnings.filterwarnings("ignore")

        # Suppress absl/mediapipe logging
        logging.getLogger("absl").setLevel(logging.ERROR)
        sys.path.append("/root/src")
        sys.path.append("/root/SegRefiner")

        from model_loader import model_segmentation, model_refiner, model_bbox, model_keypoints
        from segmentation_mp import MediaPipeSkinSegmenter
        from segrefiner import SegRefinerWrapper
        from bbox import DetectNeck
        from keypoints import DetectKeypoints

        segmentation_path = model_segmentation()
        refiner_path      = model_refiner()
        bbox_path         = model_bbox()
        keypoints_path    = model_keypoints()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("Load Model")
        self.mediapipe  = MediaPipeSkinSegmenter(segmentation_path)
        self.segrefiner = SegRefinerWrapper(
            config_path     = "/root/SegRefiner/configs/segrefiner/segrefiner_hr.py",
            checkpoint_path = refiner_path,
            device          = device,
            target_size     = 640
        )
        self.bbox      = DetectNeck(bbox_path)
        self.keypoints = DetectKeypoints(keypoints_path)

        # Warm up the GPU with a dummy forward pass to avoid cold-start latency
        self.logger.info("Warming Up GPU")
        dummy_img  = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_mask = np.zeros((640, 640),    dtype=np.uint8)
        self.segrefiner.run(dummy_img, dummy_mask, iterations=1)
        self.logger.info("GPU warm up done!")

    def analyze(self, image_bytes: bytes) -> dict:
        """
        Run the full inference pipeline on raw image bytes and return
        neck/shoulder keypoints and mask data as a JSON-compatible dict.

        Full inference pipeline:
          1. Decode image bytes to a BGR NumPy array.
          2. Crop the upper-body region and resize to the model input resolution.
          3. Run MediaPipe segmentation to obtain coarse skin/neck masks.
          4. Refine mask boundaries with SegRefiner.
          5. Blend MediaPipe and SegRefiner masks for the final mask.
          6. Check shoulder visibility via hair-region ratio.
          7. Detect the neck bounding box with the YOLO model.
          8. Detect neck/shoulder keypoints with HRNet inside the bounding box.
          9. Snap keypoints to the nearest mask contour for sub-pixel accuracy.
          10. Map refined keypoints back to the original image coordinate space.
          11. Map mask coordinates back to the original image coordinate space.

        Args:
            image_bytes: Raw bytes of the uploaded image file.

        Raises:
            ValueError: If the image bytes cannot be decoded by OpenCV.
            ShoulderNeckNotVisible: If hair occludes the shoulder area, or if
                                    keypoint confidence scores are below threshold.

        Returns:
            dict: JSON-compatible dict containing:
              - id (str): UUID identifying this inference result.
              - right_neck_shoulder_point (dict):
                  - original_point (list[int]): [x, y] in the original image space.
                  - resized_point  (list[int]): [x, y] in the resized image space.
                  - score          (float)    : Detection confidence score.
              - left_neck_shoulder_point (dict):
                  - original_point (list[int]): [x, y] in the original image space.
                  - resized_point  (list[int]): [x, y] in the resized image space.
                  - score          (float)    : Detection confidence score.
              - mask_area (list[list[int]]): Contour coordinates mapped to the original image space.
              - img_size  (list[int])      : [width, height] of the original image.
        """
        import cv2
        import numpy as np
        import uuid
        import sys
        sys.path.append("/root/src")
        from preprocess import snap_to_nearest_contour, boundary_blend, mask_to_coords, to_json, hair_detection
        from resize_image import resize, crop_upper_body, map_keypoint_to_original
        from exception import ShoulderNeckNotVisible

        time_start = time.time()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")
        h, w = img.shape[:2]

        face, face_info = crop_upper_body(img)
        image, r, top, left = resize(face)

        mask, image_seg, skin_mask, neck_face, neck_body = self.mediapipe.segment(image)
        mask_segrefiner = self.segrefiner.run(image, mask, iterations=1)

        mask_final     = boundary_blend(mask,      mask_segrefiner, boundary_width=3)
        neckface_final = boundary_blend(neck_face, mask_segrefiner, boundary_width=3)
        neckbody_final = boundary_blend(neck_body, mask_segrefiner, boundary_width=3)
        mask_coordinate = mask_to_coords(neckbody_final)

        # Check shoulder visibility — low ratio means hair is occluding the area
        shoulder_image_ratio = hair_detection(mask_coordinate, left)
        if shoulder_image_ratio < 0.6:
            raise ShoulderNeckNotVisible(cause="Hair may be covering the shoulder area")

        bbox, bbox_coords, _ = self.bbox.detect_bbox(image)
        bx1, by1, bx2, by2  = bbox_coords

        keypoints            = self.keypoints.detect_kp(bbox)
        keypoints_on_resized = [(int(x) + bx1, int(y) + by1, float(score)) for x, y, score in keypoints]
        keypoints_snapped    = snap_to_nearest_contour(neckface_final, mask_final, keypoints_on_resized)

        rx, ry, rscore, rval = keypoints_snapped[0]
        lx, ly, lscore, lval = keypoints_snapped[1]

        # Low confidence means the neck/shoulder landmarks are not clearly visible
        if rscore < 0.35 or lscore < 0.35:
            raise ShoulderNeckNotVisible(
                cause=f"Neck-shoulder area not clearly visible, right-shoulder score {rscore}, left-shoulder score {lscore}"
            )

        keypoints_original = [
            map_keypoint_to_original(x, y, r, top, left, face_info)
            for x, y, _, _ in keypoints_snapped
        ]
        mask_coordinate_original = [
            list(map_keypoint_to_original(x, y, r, top, left, face_info))
            for x, y in mask_coordinate
        ]

        time_total = time.time() - time_start
        self.logger.info(f"Total Time Process Model: {time_total}")

        return {
            "id": to_json(str(uuid.uuid4())),
            "right_neck_shoulder_point": {
                "original_point": to_json(list(keypoints_original[0])),
                "resized_point" : to_json([rx, ry]),
                "score"         : to_json(rscore),
            },
            "left_neck_shoulder_point": {
                "original_point": to_json(list(keypoints_original[1])),
                "resized_point" : to_json([lx, ly]),
                "score"         : to_json(lscore),
            },
            "mask_area": to_json(mask_coordinate_original),
            "img_size" : to_json([w, h]),
        }

    def visualize(self, image_bytes: bytes) -> bytes:
        """
        Run the full inference pipeline on raw image bytes and return a JPEG
        with detected keypoints drawn on it, for visual debugging and inspection.

        Executes the same pipeline as ``analyze`` (segmentation → bounding box →
        keypoint detection → contour snapping) and then renders the two snapped
        neck/shoulder keypoints as green circles directly on the resized image.

        Args:
            image_bytes: Raw bytes of the uploaded image file.

        Raises:
            ValueError: If the image bytes cannot be decoded by OpenCV.
            ShoulderNeckNotVisible: If hair occludes the shoulder area, or if
                                    keypoint confidence scores are below threshold.

        Returns:
            bytes: JPEG-encoded image bytes with the two snapped keypoints
                   overlaid as filled green circles (radius 2 px).
        """
        import cv2
        import numpy as np
        import sys
        sys.path.append("/root/src")
        from preprocess import snap_to_nearest_contour, boundary_blend, mask_to_coords, hair_detection
        from resize_image import resize, crop_upper_body
        from exception import ShoulderNeckNotVisible

        time_start = time.time()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")

        face, face_info      = crop_upper_body(img)
        image, r, top, left  = resize(face)

        mask, image_seg, skin_mask, neck_face, neck_body = self.mediapipe.segment(image)
        mask_segrefiner = self.segrefiner.run(image, mask, iterations=1)
        mask_final      = boundary_blend(mask,      mask_segrefiner, boundary_width=3)
        neckface_final  = boundary_blend(neck_face, mask_segrefiner, boundary_width=3)
        neckbody_final  = boundary_blend(neck_body, mask_segrefiner, boundary_width=3)
        mask_coordinate = mask_to_coords(neckbody_final)

        # Check shoulder visibility — low ratio means hair is occluding the area
        shoulder_image_ratio = hair_detection(mask_coordinate, left)
        if shoulder_image_ratio < 0.6:
            raise ShoulderNeckNotVisible(cause="Hair may be covering the shoulder area")

        bbox, bbox_coords, score = self.bbox.detect_bbox(image)
        bx1, by1, bx2, by2      = bbox_coords

        keypoints            = self.keypoints.detect_kp(bbox)
        keypoints_on_resized = [(int(x) + bx1, int(y) + by1, float(score)) for x, y, score in keypoints]
        keypoints_snapped    = snap_to_nearest_contour(neckface_final, mask_final, keypoints_on_resized)

        rx, ry, rscore, rval = keypoints_snapped[0]
        lx, ly, lscore, lval = keypoints_snapped[1]

        # Low confidence means the neck/shoulder landmarks are not clearly visible
        if rscore < 0.35 or lscore < 0.35:
            raise ShoulderNeckNotVisible(
                cause=f"Neck-shoulder area not clearly visible, right-shoulder score {rscore}, left-shoulder score {lscore}"
            )

        cv2.circle(image, (rx, ry), 2, (0, 255, 0), -1)
        cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)

        _, buffer  = cv2.imencode(".jpg", image)
        time_total = time.time() - time_start
        self.logger.info(f"Total Time Process Model: {time_total}")

        return buffer.tobytes()

    @modal.asgi_app()
    def endpoint(self):
        """
        Build and return the FastAPI ASGI application served by Modal.

        Registers exception handlers for all custom exception types and mounts
        the following routes:

          - GET  /          : Root health-check.
          - POST /analyze   : Full inference pipeline returning a JSON result.
          - POST /visualize : Full inference pipeline returning an annotated JPEG.
          - GET  /health    : Lightweight health-check for load balancers.

        Returns:
            FastAPI: Configured ASGI application instance.
        """
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse, StreamingResponse
        from exception import ModelLoadError, DetectionFailed, ShoulderNeckNotVisible
        import io

        api_app = FastAPI()

        @api_app.exception_handler(ModelLoadError)
        async def model_load_handler(request, exc):
            """
            Handle ModelLoadError exceptions and return a JSON error response.

            Returns:
                JSONResponse with error_code ``MODEL_LOAD_ERROR`` and the exception message.
            """
            self.logger.error(str(exc))
            return JSONResponse(
                status_code=exc.status_code,
                content={"error_code": "MODEL_LOAD_ERROR", "message": str(exc)}
            )

        @api_app.exception_handler(DetectionFailed)
        async def detection_failed_handler(request, exc):
            """
            Handle DetectionFailed exceptions and return a JSON error response.

            Returns:
                JSONResponse with error_code ``DETECTION_FAILED`` and the exception message.
            """
            self.logger.error(str(exc))
            return JSONResponse(
                status_code=exc.status_code,
                content={"error_code": "DETECTION_FAILED", "message": str(exc)}
            )

        @api_app.exception_handler(ShoulderNeckNotVisible)
        async def shoulder_not_visible_handler(request, exc):
            """
            Handle ShoulderNeckNotVisible exceptions and return a JSON error response.

            Returns:
                JSONResponse with error_code ``SHOULDER_NECK_NOT_VISIBLE`` and the exception message.
            """
            self.logger.error(str(exc))
            return JSONResponse(
                status_code=exc.status_code,
                content={"error_code": "SHOULDER_NECK_NOT_VISIBLE", "message": str(exc)}
            )

        @api_app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            """
            Handle ValueError exceptions, typically raised for invalid or unreadable images.

            Returns:
                JSONResponse (HTTP 400) with error_code ``INVALID_IMAGE`` and the exception message.
            """
            self.logger.error(f"Value error: {exc}")
            return JSONResponse(
                status_code=400,
                content={"error_code": "INVALID_IMAGE", "message": str(exc)}
            )

        @api_app.exception_handler(Exception)
        async def unexpected_handler(request, exc):
            """
            Catch-all handler for any unhandled exceptions.

            Returns:
                JSONResponse (HTTP 500) with error_code ``INTERNAL_ERROR`` and the exception message.
            """
            self.logger.error(f"Unexpected: {exc}")
            return JSONResponse(
                status_code=500,
                content={"error_code": "INTERNAL_ERROR", "message": str(exc)}
            )

        @api_app.get("/")
        async def root():
            """
            Root health-check endpoint.

            Returns:
                dict: A simple status message confirming the server is running.
            """
            return {"message": "Server is running"}

        @api_app.post("/analyze")
        async def analyze(file: UploadFile = File(...)):
            """
            Analyze an uploaded portrait image and return neck/shoulder keypoints and mask data.

            Args:
                file: Uploaded image file (multipart/form-data).

            Returns:
                dict: JSON result from ``DetectNecklacePoint.analyze``.
            """
            time_start = time.time()
            self.logger.info(f"Analyze request: {file.filename}")
            image_bytes = await file.read()
            self.logger.info(f"Image Size: {len(image_bytes)/1024:.1f}KB")
            result = self.analyze(image_bytes)
            self.logger.info(f"Analyze success: {file.filename} | Total time: {time.time()-time_start:.3f}s")
            self.logger.info(f"JSON Response: {result}")
            return result

        @api_app.post("/visualize")
        async def visualize(file: UploadFile = File(...)):
            """
            Run inference on an uploaded image and return a JPEG with keypoints drawn on it.

            Args:
                file: Uploaded image file (multipart/form-data).

            Returns:
                StreamingResponse: JPEG image with snapped keypoints overlaid.
            """
            time_start = time.time()
            self.logger.info(f"Visualize request: {file.filename}")
            image_bytes = await file.read()
            self.logger.info(f"Image Size: {len(image_bytes)/1024:.1f}KB")
            result = self.visualize(image_bytes)
            self.logger.info(f"Visualize success: {file.filename} | Total time: {time.time()-time_start:.3f}s")
            return StreamingResponse(io.BytesIO(result), media_type="image/jpeg")

        @api_app.get("/health")
        def health():
            """
            Lightweight health-check endpoint for load balancers and monitoring tools.

            Returns:
                dict: ``{"status": "ok"}`` when the server is running normally.
            """
            return {"status": "ok"}

        return api_app
