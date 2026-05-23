import cv2
import numpy as np
import mediapipe as mp
from exception import DetectionFailed,ModelLoadError

class MediaPipeSkinSegmenter:
    def __init__(self, model_path):
        """Initialize MediaPipe image segmenter from a local model file."""
        try:
            BaseOptions = mp.tasks.BaseOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            self.segmenter = ImageSegmenter.create_from_options(
                ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    output_category_mask=True
                )
            )
            self._last_error = None
        except Exception as e:
            raise ModelLoadError(method="MediaPipe Segmenter", cause=str(e))

    def predict_mask(self, image: np.ndarray):
        """
        Run segmentation and return a category mask resized to match the input image.

        Returns:
            category_mask: HxW array where each pixel holds a category index (int)
        """
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.segmenter.segment(mp_image)
            mask = result.category_mask.numpy_view()
            return cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        except Exception as e:
            self._last_error = str(e)
            return None

    def segment(self, image: np.ndarray):
        """
        Produce a binary skin mask and a masked overlay from the input image.

        MediaPipe category indices used:
            2 = hair, 3 = body skin, 4 = face skin

        Args:
            image: RGB image as a numpy array (HxWx3)

        Returns:
            skin_mask:   Binary mask (0 or 255) of skin/hair regions after noise removal
            overlay:     Image with non-skin pixels replaced by gray (128)
            mask_person: Raw binary mask before morphological cleaning
        """
        category_mask = self.predict_mask(image)
        if category_mask is None:
            raise DetectionFailed(method="Mediapipe", cause=self._last_error)
            
        mask_person = np.isin(category_mask, [2,3,4]).astype(np.uint8) * 255
        mask_neckface= np.isin(category_mask, [2,3]).astype(np.uint8) * 255
        mask_neckbody = np.isin(category_mask, [2,4]).astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)                                     
        skin_mask = cv2.morphologyEx(mask_person, cv2.MORPH_OPEN, kernel)
        neckface_mask= cv2.morphologyEx(mask_neckface, cv2.MORPH_OPEN, kernel)
        neckbody_mask= cv2.morphologyEx(mask_neckbody, cv2.MORPH_OPEN, kernel)
        mask_3c = np.stack([skin_mask] * 3, axis=-1)
        overlay = np.where(mask_3c == 255, image, 128).astype(np.uint8)

        return skin_mask, overlay,mask_person,neckface_mask,neckbody_mask