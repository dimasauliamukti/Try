import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from mmcv import Config
from mmcv.runner import load_checkpoint
import mmdet.models
from mmdet.models.builder import DETECTORS, build_detector
from mmdet.core.mask import BitmapMasks 
from exception import ModelLoadError, DetectionFailed


IMAGENET_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
IMAGENET_STD  = np.array([58.395,  57.12,   57.375], dtype=np.float32)


class SegRefinerWrapper:
    """
    SegRefiner-based mask refinement wrapper.

    Loads a semantic segmentation refinement model via MMDetection and
    iteratively refines coarse binary masks using a diffusion-based
    SegRefiner architecture.

    Raises:
        ModelLoadError: If the config file or model checkpoint fails to load.
        DetectionFailed: If mask refinement inference fails during ``run``.
    """

    def __init__(self, config_path: str, checkpoint_path: str,
                 device: str = "cuda:0", target_size: int = 512):
        """
        Initialise the SegRefiner model from a config file and checkpoint.

        Sets deterministic seeds for reproducibility, loads the MMDetection
        config, overrides training settings for inference mode, builds the
        detector, and moves it to the target device.

        Args:
            config_path:     Path to the MMDetection ``.py`` config file.
            checkpoint_path: Path to the model ``.pth`` checkpoint file.
            device:          Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
            target_size:     Spatial resolution to which inputs are resized
                             before inference (default: 512).

        Raises:
            ModelLoadError: If the config file cannot be parsed, or if the
                            checkpoint cannot be loaded onto the model.
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        self.device      = device
        self.target_size = target_size
        try:
            cfg = Config.fromfile(config_path)
        except Exception as e:
            raise ModelLoadError(method="SegRefiner Config", cause=str(e))
        cfg.model.train_cfg = None
        cfg.model.type = 'SegRefinerSemantic'
        cfg.model.task = 'semantic'

        cfg.model.test_cfg = dict(
            model_size    = 256,
            fine_prob_thr = 0.9,
            iou_thr       = 0.3,
            batch_max     = 32
        )
        try:
            self.model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
            load_checkpoint(self.model, checkpoint_path, map_location="cpu", strict=False)
            self.model.eval()
            self.model.to(device)
        except Exception as e:
            raise ModelLoadError(method="SegRefiner Model", cause=str(e))
        self.dev = next(self.model.parameters()).device
        self._last_error = None

    def preprocess(self, image_rgb: np.ndarray, coarse_mask: np.ndarray) -> tuple:
        """
        Prepare the image and coarse mask for SegRefiner inference.

        Resizes both the image and mask to ``target_size``, normalises the
        image using ImageNet statistics, converts them to the tensor and
        BitmapMasks formats expected by MMDetection, and constructs the
        image meta dictionary required by the model.

        Args:
            image_rgb:   RGB image array (H x W x 3, uint8).
            coarse_mask: Grayscale binary mask (H x W, uint8) where pixel
                         values above 127 are treated as foreground.

        Returns:
            A four-element tuple ``(img_tensor, coarse_masks, meta, ori_size)``
            where:
              - img_tensor    (torch.Tensor): Normalised image tensor of shape
                                             (1, 3, target_size, target_size).
              - coarse_masks  (list[BitmapMasks]): Single-element list containing
                                                   the resized mask as a BitmapMasks
                                                   object.
              - meta          (list[dict]): Single-element list of image metadata
                                           dicts consumed by MMDetection.
              - ori_size      (tuple[int, int]): Original ``(H, W)`` of the input
                                                 image, used to restore output
                                                 resolution in ``postprocess``.
        """
        h, w = image_rgb.shape[:2]
        ts   = self.target_size

        mask_binary   = (coarse_mask > 127).astype(np.float32)
        image_resized = cv2.resize(image_rgb, (ts, ts), interpolation=cv2.INTER_LINEAR)
        mask_resized  = cv2.resize(mask_binary, (ts, ts), interpolation=cv2.INTER_NEAREST)
        image_norm    = (image_resized.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor    = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()

        mask_np      = mask_resized[np.newaxis, :, :]
        bitmap_mask  = BitmapMasks(mask_np, ts, ts)
        coarse_masks = [bitmap_mask]

        meta = [{
            "filename"    : "input",
            "ori_filename": "input/im.jpg",
            "ori_shape"   : (ts, ts, 3),
            "img_shape"   : (ts, ts, 3),
            "pad_shape"   : (ts, ts, 3),
            "scale_factor": 1.0,
            "flip"        : False,
            "img_norm_cfg": {
                "mean"  : IMAGENET_MEAN.tolist(),
                "std"   : IMAGENET_STD.tolist(),
                "to_rgb": True
            }
        }]

        return img_tensor, coarse_masks, meta, (h, w)

    def postprocess(self, result, original_size: tuple) -> np.ndarray:
        """
        Convert raw model output back to a full-resolution binary mask.

        Extracts the first predicted mask from the model result, thresholds
        it at 0.5 to produce a binary uint8 array, then rescales it to the
        original image dimensions using nearest-neighbour interpolation.

        Args:
            result:        Raw output returned by ``simple_test_semantic``.
                           Expected to be a nested sequence whose first element
                           contains the predicted probability mask.
            original_size: ``(H, W)`` tuple specifying the target output
                           resolution.

        Returns:
            Binary mask (np.ndarray): H x W uint8 array with values 0 or 255,
            resized to match ``original_size``.
        """
        h, w = original_size

        mask_np = result[0][0]
        if isinstance(mask_np, torch.Tensor):
            mask_np = mask_np.cpu().numpy()

        mask_np     = mask_np.astype(np.float32)
        mask_binary = (mask_np >= 0.5).astype(np.uint8) * 255
        return cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

    def refine(self, image_rgb: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray | None:
        """
        Run a single SegRefiner inference pass to refine a coarse mask.

        Calls ``preprocess`` to prepare inputs, runs ``simple_test_semantic``
        under ``torch.no_grad()``, then calls ``postprocess`` to recover a
        full-resolution binary mask.

        Args:
            image_rgb:   RGB image array (H x W x 3, uint8).
            coarse_mask: Grayscale binary mask (H x W, uint8) to be refined.

        Returns:
            Refined binary mask (np.ndarray): H x W uint8 array with values
            0 or 255, or ``None`` if inference raises an exception (the
            exception message is stored in ``self._last_error``).
        """
        try:
            img_tensor, coarse_masks, meta, ori_size = self.preprocess(image_rgb, coarse_mask)
            with torch.no_grad():
                result = self.model.simple_test_semantic(
                    meta,
                    img=img_tensor.to(self.dev),
                    coarse_masks=coarse_masks
                )
            return self.postprocess(result, ori_size)
        except Exception as e:
            self._last_error = str(e)
            return None

    def run(self, image: np.ndarray, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Refine a coarse mask and optionally dilate the result.

        Converts the input BGR image to RGB, calls ``refine`` to obtain the
        SegRefiner output mask, then applies morphological dilation with a
        2×2 kernel for the specified number of iterations to slightly expand
        the mask boundary.

        Args:
            image:      BGR image array (H x W x 3, uint8) as loaded by OpenCV.
            mask:       Grayscale binary mask (H x W, uint8) to be refined.
            iterations: Number of dilation iterations applied after refinement.
                        Set to 0 to skip dilation entirely (default: 1).

        Raises:
            DetectionFailed: If ``refine`` returns None.

        Returns:
            Refined and optionally dilated binary mask (np.ndarray):
            H x W uint8 array with values 0 or 255.
        """
        image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        refined_mask = self.refine(image_rgb, mask)
        if refined_mask is None:
            raise DetectionFailed(method="SegRefiner", cause=self._last_error)

        if iterations > 0:
            kernel       = np.ones((2, 2), np.uint8)
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=iterations)

        return refined_mask