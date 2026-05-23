import cv2
import numpy as np
import mediapipe as mp
from exception import DetectionFailed, ModelLoadError


class MediaPipeSkinSegmenter:
    """
    MediaPipe-based skin and neck segmentation wrapper.

    Loads a multi-class segmentation TFLite model and produces binary masks
    for the skin, neck-face, and neck-body regions using MediaPipe category
    indices.

    Category indices used (selfie_multiclass):
      - 0 = background
      - 1 = hair
      - 2 = body skin
      - 3 = face skin
      - 4 = clothes
      - 5 = others (accessories)

    Raises:
        ModelLoadError: If the MediaPipe ImageSegmenter fails to initialise.
        DetectionFailed: If segmentation inference fails during ``segment``.
    """

    def __init__(self, model_path: str) :
        """
        Initialise the MediaPipe ImageSegmenter from a local TFLite model file.

        Args:
            model_path: Path to the ``selfie_multiclass.tflite`` model file.

        Raises:
            ModelLoadError: If the segmenter cannot be created from the given model path.
        """
        try:
            BaseOptions           = mp.tasks.BaseOptions
            ImageSegmenter        = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode

            self.segmenter = ImageSegmenter.create_from_options(
                ImageSegmenterOptions(
                    base_options        = BaseOptions(model_asset_path=model_path),
                    running_mode        = VisionRunningMode.IMAGE,
                    output_category_mask= True
                )
            )
            self._last_error = None
        except Exception as e:
            raise ModelLoadError(method="MediaPipe Segmenter", cause=str(e))

    def predict_mask(self, image: np.ndarray):
        """
        Run the MediaPipe segmenter and return a per-pixel category mask.

        Converts the input array to a MediaPipe Image, runs single-image
        segmentation, and resizes the resulting category mask back to the
        original image resolution using nearest-neighbour interpolation.

        Args:
            image: RGB image array (H x W x 3, uint8).

        Returns:
            category_mask (np.ndarray): H x W uint8 array where each pixel holds
                                        a MediaPipe category index, resized to
                                        match the input image dimensions.
                                        Returns None if inference fails.
        """
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result   = self.segmenter.segment(mp_image)
            mask     = result.category_mask.numpy_view()
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
        Produce binary skin and neck masks along with a masked overlay image.

        Calls ``predict_mask`` to obtain the raw category mask, then derives
        three region masks by combining relevant category indices. Morphological
        opening with a 5×5 kernel is applied to each mask to remove small noise
        regions before returning.

        Region definitions:
          - skin mask      : categories 2 (body skin) + 3 (face skin) + 4 (clothes)
          - neck-face mask : categories 2 (body skin) + 3 (face skin)
          - neck-body mask : categories 2 (body skin) + 4 (clothes)

        Args:
            image: RGB image array (H x W x 3, uint8).

        Raises:
            DetectionFailed: If ``predict_mask`` returns None.

        Returns:
            (skin_mask, overlay, mask_person, neckface_mask, neckbody_mask) where:
              - skin_mask     (np.ndarray): Binary mask (0/255) of all skin/clothes regions
                                           after morphological noise removal.
              - overlay       (np.ndarray): Input image with non-skin pixels replaced
                                           by gray (128), same shape as ``image``.
              - mask_person   (np.ndarray): Raw binary skin mask before morphological
                                           cleaning, combining body skin + face skin + clothes.
              - neckface_mask (np.ndarray): Cleaned binary mask for the neck-face region
                                            (body skin + face skin).
              - neckbody_mask (np.ndarray): Cleaned binary mask for the neck-body region
                                            (body skin + clothes).
        """
        category_mask = self.predict_mask(image)
        if category_mask is None:
            raise DetectionFailed(method="Mediapipe", cause=self._last_error)

        mask_person   = np.isin(category_mask, [2, 3, 4]).astype(np.uint8) * 255
        mask_neckface = np.isin(category_mask, [2, 3]).astype(np.uint8)    * 255
        mask_neckbody = np.isin(category_mask, [2, 4]).astype(np.uint8)    * 255

        kernel       = np.ones((5, 5), np.uint8)
        skin_mask    = cv2.morphologyEx(mask_person,   cv2.MORPH_OPEN, kernel)
        neckface_mask= cv2.morphologyEx(mask_neckface, cv2.MORPH_OPEN, kernel)
        neckbody_mask= cv2.morphologyEx(mask_neckbody, cv2.MORPH_OPEN, kernel)

        mask_3c = np.stack([skin_mask] * 3, axis=-1)
        overlay = np.where(mask_3c == 255, image, 128).astype(np.uint8)

        return skin_mask, overlay, mask_person, neckface_mask, neckbody_mask