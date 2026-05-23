import onnxruntime as ort
import numpy as np
import cv2
from exception import DetectionFailed, ModelLoadError


class DetectKeypoints:
    """
    HRNet-based neck/shoulder keypoint detector backed by an ONNX runtime session.

    Runs a full inference pipeline consisting of affine-aligned pre-processing,
    ONNX model inference, and heatmap decoding to produce (x, y, confidence)
    keypoint tuples in the original image coordinate space.

    Raises:
        ModelLoadError: If the ONNX session cannot be initialised at construction.
        DetectionFailed: If inference or decoding fails during ``detect_kp``.
    """

    def __init__(self, model_path: str) -> None:
        """
        Load the HRNet ONNX model and initialise the inference session.

        Args:
            model_path: Path to the HRNet keypoint detection ONNX model file.

        Raises:
            ModelLoadError: If the ONNX InferenceSession fails to initialise.
        """
        try:
            self.model = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception as e:
            raise ModelLoadError(method="Keypoints Models", cause=str(e))

    def get_affine_transform_udp(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        rot: float,
        output_size: tuple,
    ) -> np.ndarray:
        """
        Compute a 2×3 affine transformation matrix using the UDP (Unbiased
        Data Processing) convention used by HRNet pre-processing.

        Constructs source and destination triangles from the image center, scale,
        and rotation angle, then calls ``cv2.getAffineTransform`` to derive the
        matrix that maps source pixels to the model input grid.

        Args:
            center:      (x, y) centre of the source region in pixel coordinates.
            scale:       (width, height) of the source region in pixels.
            rot:         Rotation angle in degrees (0 for no rotation).
            output_size: (width, height) of the model input resolution.

        Returns:
            M (np.ndarray): 2×3 float32 affine transformation matrix.
        """
        src_w        = scale[0]
        dst_w, dst_h = output_size[0], output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)

        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5])
        dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

        src[2, :] = src[1, :] + np.array([-src_dir[1], src_dir[0]])
        dst[2, :] = dst[1, :] + np.array([-dst_dir[1], dst_dir[0]])

        M = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        return M

    def preprocess_onnx(
        self,
        img_bgr: np.ndarray,
        input_size: tuple = (320, 128),
    ) -> tuple:
        """
        Pre-process a BGR crop into a normalised BCHW tensor aligned to the
        model input size via an affine warp.

        Steps:
          1. Compute a scale that preserves the aspect ratio of the input image
             relative to the model input resolution.
          2. Build a UDP affine matrix mapping the source region to the model grid.
          3. Warp the image with bilinear interpolation.
          4. Convert to RGB, subtract ImageNet mean, and divide by std.
          5. Reshape to (1, C, H, W) for ONNX input.

        Args:
            img_bgr:    BGR input image crop (H x W x 3, uint8).
            input_size: (width, height) of the model input resolution.
                        Defaults to (288, 128).

        Returns:
            (blob, M) where:
              - blob (np.ndarray): Float32 BCHW tensor of shape (1, 3, H, W).
              - M    (np.ndarray): 2×3 affine matrix used for the warp, needed
                                   by ``decode_heatmap`` to invert the transform.
        """
        orig_h, orig_w = img_bgr.shape[:2]

        center       = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
        aspect_ratio = input_size[0] / input_size[1]

        if orig_w > orig_h * aspect_ratio:
            scale = np.array([orig_w, orig_w / aspect_ratio], dtype=np.float32)
        else:
            scale = np.array([orig_h * aspect_ratio, orig_h], dtype=np.float32)

        M = self.get_affine_transform_udp(center, scale, 0, input_size)

        img_warped = cv2.warpAffine(
            img_bgr, M,
            (input_size[0], input_size[1]),
            flags=cv2.INTER_LINEAR
        )

        img  = cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
        std  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
        img  = (img - mean) / std

        blob = img.transpose(2, 0, 1)[np.newaxis]  # BCHW
        return blob, M

    def decode_heatmap(
        self,
        heatmap: np.ndarray,
        M: np.ndarray,
        hm_w: int   = 80,
        hm_h: int   = 32,
        input_size: tuple = (320, 128),
    ) -> list:
        """
        Decode the model's output heatmaps into (x, y, confidence) keypoints
        in the original image coordinate space.

        For each keypoint channel the peak location in the heatmap is found,
        scaled from heatmap resolution to model input resolution, then mapped
        back to the original image space by inverting the affine transform.

        Args:
            heatmap:    Raw model output of shape (1, num_keypoints, hm_h, hm_w).
            M:          2×3 affine matrix returned by ``preprocess_onnx``.
            hm_w:       Width of the heatmap grid. Defaults to 72.
            hm_h:       Height of the heatmap grid. Defaults to 32.
            input_size: (width, height) of the model input resolution used to
                        compute the heatmap-to-input scale factors.
                        Defaults to (288, 128).

        Returns:
            List of (x, y, confidence) tuples (one per keypoint channel) where:
              - x, y      (int)  : Keypoint coordinates in the original image space.
              - confidence (float): Peak heatmap value used as the detection score.
        """
        scale_x = input_size[0] / hm_w  # 288 / 72 = 4.0
        scale_y = input_size[1] / hm_h

        M_inv = cv2.invertAffineTransform(M)

        keypoints = []
        for kp_idx in range(heatmap.shape[1]):
            hm          = heatmap[0, kp_idx]
            idx         = np.unravel_index(np.argmax(hm), hm.shape)
            y_hm, x_hm = int(idx[0]), int(idx[1])

            # Scale peak location from heatmap resolution to model input resolution
            x_input = x_hm * scale_x
            y_input = y_hm * scale_y

            # Apply inverse affine transform to get original image coordinates
            x_coord = M_inv[0, 0] * x_input + M_inv[0, 1] * y_input + M_inv[0, 2]
            y_coord = M_inv[1, 0] * x_input + M_inv[1, 1] * y_input + M_inv[1, 2]

            conf = float(hm.max())
            keypoints.append((int(x_coord), int(y_coord), conf))

        return keypoints

    def detect_kp(self, image: np.ndarray) -> list:
        """
        Run the full keypoint detection pipeline on a BGR image crop.

        Sequentially calls ``preprocess_onnx``, runs the ONNX model, and
        decodes the output heatmaps via ``decode_heatmap``.

        Args:
            image: BGR image crop containing the neck/shoulder region (uint8).

        Raises:
            DetectionFailed: If inference raises any exception, or if fewer
                             than 2 keypoints are returned by the model.

        Returns:
            List of at least 2 (x, y, confidence) tuples representing the
            detected neck/shoulder keypoints in the input image coordinate space.
        """
        try:
            input_name = self.model.get_inputs()[0].name
            blob, M    = self.preprocess_onnx(image)
            heatmap    = self.model.run(None, {input_name: blob})[0]
            keypoints  = self.decode_heatmap(heatmap, M)

            if len(keypoints) < 2:
                raise ValueError(f"Keypoints only detected {len(keypoints)}")

            return keypoints
        except Exception as e:
            raise DetectionFailed(method="Keypoints Detection", cause=str(e))
        
        
        
#         import onnxruntime as ort
# import numpy as np
# import cv2
# from exception import DetectionFailed,ModelLoadError


# class DetectKeypoints():
#     def __init__(self,model_path):
#         try:
#             self.model=ort.InferenceSession(model_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
#         except Exception as e:
#             raise ModelLoadError(method="Keypoints Models", cause=str(e))        

#     def get_affine_transform_udp(self,center, scale, rot, output_size):
#         src_w        = scale[0]
#         dst_w, dst_h = output_size[0], output_size[1]

#         rot_rad = np.pi * rot / 180
#         src_dir = np.array([0, src_w * -0.5])
#         dst_dir = np.array([0, dst_w * -0.5])

#         src = np.zeros((3, 2), dtype=np.float32)
#         dst = np.zeros((3, 2), dtype=np.float32)

#         src[0, :] = center
#         src[1, :] = center + src_dir
#         dst[0, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5])
#         dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

#         src[2, :] = src[1, :] + np.array([-src_dir[1], src_dir[0]])
#         dst[2, :] = dst[1, :] + np.array([-dst_dir[1], dst_dir[0]])

#         M = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
#         return M


#     # def preprocess_onnx(self,img_bgr, input_size=(320, 128)):
#     def preprocess_onnx(self, img_bgr, input_size=(288, 128)): 

#         orig_h, orig_w = img_bgr.shape[:2]

#         center       = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
#         aspect_ratio = input_size[0] / input_size[1]

#         if orig_w > orig_h * aspect_ratio:
#             scale = np.array([orig_w, orig_w / aspect_ratio], dtype=np.float32)
#         else:
#             scale = np.array([orig_h * aspect_ratio, orig_h], dtype=np.float32)

#         M = self.get_affine_transform_udp(center, scale, 0, input_size)

#         img_warped = cv2.warpAffine(
#             img_bgr, M,
#             (input_size[0], input_size[1]),
#             flags=cv2.INTER_LINEAR
#         )

#         img  = cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB).astype(np.float32)
#         mean = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
#         std  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
#         img  = (img - mean) / std

#         blob = img.transpose(2, 0, 1)[np.newaxis]   # BCHW
#         return blob, M


#     def decode_heatmap(self, heatmap, M,
#                 hm_w=72, hm_h=32,          # ← hm_w: 80 → 72
#                 input_size=(288, 128)):     # ← 320 → 288
#         scale_x = input_size[0] / hm_w   # 288 / 72 = 4.0 ✅
#         scale_y = input_size[1] / hm_h 

#         M_inv = cv2.invertAffineTransform(M)

#         keypoints = []
#         for kp_idx in range(heatmap.shape[1]):
#             hm        = heatmap[0, kp_idx]
#             idx       = np.unravel_index(np.argmax(hm), hm.shape)
#             y_hm, x_hm = int(idx[0]), int(idx[1])

#             x_input = x_hm * scale_x
#             y_input = y_hm * scale_y

#             x_coord = M_inv[0, 0] * x_input + M_inv[0, 1] * y_input + M_inv[0, 2]
#             y_coord = M_inv[1, 0] * x_input + M_inv[1, 1] * y_input + M_inv[1, 2]

#             conf = float(hm.max())
#             # int() di sini — sama persis dengan int(kp[0]) di PTH
#             keypoints.append((int(x_coord), int(y_coord), conf))

#         return keypoints
    
#     def detect_kp(self, image):
#         try:
#             input_name = self.model.get_inputs()[0].name
#             blob, M = self.preprocess_onnx(image)
#             heatmap = self.model.run(None, {input_name: blob})[0]
#             keypoints = self.decode_heatmap(heatmap, M)
            
#             if len(keypoints) < 2:
#                 raise ValueError(f"Keypoints only detected {len(keypoints)}")
            
#             return keypoints
#         except Exception as e:
#             raise DetectionFailed(method="Keypoints Detection", cause=str(e))