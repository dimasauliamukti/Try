import cv2
import numpy as np


def boundary_blend(
    mask_mediapipe: np.ndarray,
    mask_segrefiner: np.ndarray,
    boundary_width: int = 15,
) -> np.ndarray:
    """
    Blend two binary masks by replacing the boundary zone of the MediaPipe
    mask with values from the SegRefiner mask, while keeping the interior
    fully white.

    Args:
        mask_mediapipe:  Binary mask from MediaPipe (uint8, 0/255).
        mask_segrefiner: Binary mask from SegRefiner (uint8, 0/255).
        boundary_width:  Half-width of the elliptical erosion/dilation kernel
                         in pixels. Defaults to 15.

    Returns:
        Blended binary mask (uint8, 0/255).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_width * 2 + 1, boundary_width * 2 + 1)
    )
    dilated       = cv2.dilate(mask_mediapipe, kernel)
    eroded        = cv2.erode(mask_mediapipe, kernel)
    boundary_zone = cv2.subtract(dilated, eroded)
    interior_zone = eroded

    mask_combined = mask_mediapipe.copy()
    mask_combined[boundary_zone > 0] = mask_segrefiner[boundary_zone > 0]
    mask_combined[interior_zone > 0] = 255
    return mask_combined


def closest_distance(
    pts: np.ndarray,
    x: float,
    y: float,
    y_tolerance: float,
    max_dist: float,
) -> np.ndarray | None:
    """
    Find the closest point in ``pts`` to (x, y) within a vertical tolerance
    band and a maximum Euclidean distance.

    Args:
        pts:         Array of (x, y) candidate points with shape (N, 2).
        x:           Query x coordinate.
        y:           Query y coordinate.
        y_tolerance: Maximum allowed vertical distance (|pt_y - y|) for a
                     point to be considered a candidate.
        max_dist:    Maximum allowed Euclidean distance from (x, y) to a
                     candidate point.

    Returns:
        The closest valid point as a 1-D array [x, y], or None if no
        candidate satisfies both constraints.
    """
    y_mask     = np.abs(pts[:, 1] - y) <= y_tolerance
    candidates = pts[y_mask]
    if len(candidates) == 0:
        return None
    dists       = np.sqrt((candidates[:, 0] - x)**2 + (candidates[:, 1] - y)**2)
    valid_mask  = dists <= max_dist
    valid_dists = dists[valid_mask]
    valid_pts   = candidates[valid_mask]
    if len(valid_pts) == 0:
        return None
    best_idx = np.argmin(valid_dists)
    return valid_pts[best_idx]


def snap_to_nearest_contour(
    neck_mask: np.ndarray,
    full_mask: np.ndarray,
    keypoints_list: list,
    y_tolerance: float = 0,
    max_dist: float    = 5,
    max_dist2: float   = 25,
) -> list:
    """
    Snap each raw keypoint to the nearest contour point on the neck or full mask.

    For each keypoint, a two-tier search is performed:
      1. Search the neck-face mask contour within ``max_dist`` pixels.
      2. If no match is found, fall back to the full mask contour within ``max_dist2`` pixels.
      3. If still no match, return the original keypoint coordinates with ``snapped=False``.

    Args:
        neck_mask:      Binary neck-face mask (uint8, 0/255) used for the primary search.
        full_mask:      Binary full-body mask (uint8, 0/255) used as fallback.
        keypoints_list: List of (x, y, score) tuples in the resized image coordinate space.
        y_tolerance:    Maximum vertical offset allowed when searching for contour candidates.
                        Defaults to 0 (exact row match).
        max_dist:       Maximum Euclidean distance for the primary neck contour search.
                        Defaults to 5.
        max_dist2:      Maximum Euclidean distance for the fallback full-mask contour search.
                        Defaults to 25.

    Returns:
        List of (x, y, score, snapped) tuples where:
          - x, y    (int)  : Snapped coordinates, or original coordinates if snapping failed.
          - score   (float): Original detection confidence score.
          - snapped (bool) : True if the point was successfully snapped to a contour.
    """
    neck_contours, _ = cv2.findContours(neck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full_contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not neck_contours or not full_contours:
        return [(x, y, score, False) for (x, y, score) in keypoints_list]

    neck_contours_pts = np.vstack([c.reshape(-1, 2) for c in neck_contours]).astype(float)
    full_contours_pts = np.vstack([c.reshape(-1, 2) for c in full_contours]).astype(float)

    results = []
    for (x, y, score) in keypoints_list:
        # Primary search — tight radius on the neck-face mask contour
        best_pt = closest_distance(neck_contours_pts, x, y, y_tolerance, max_dist)
        if best_pt is None:
            # Fallback search — wider radius on the full-body mask contour
            best_pt = closest_distance(full_contours_pts, x, y, y_tolerance, max_dist2)
        if best_pt is None:
            results.append((x, y, score, False))
        else:
            results.append((int(best_pt[0]), int(best_pt[1]), score, True))

    return results


def mask_to_coords(mask: np.ndarray, epsilon_factor: float = 0.001) -> list:
    """
    Extract and simplify the largest contour of a binary mask as a list of (x, y) coordinates.

    Finds all external contours, selects the largest by area, and applies
    Douglas-Peucker polygon approximation to reduce the number of points.

    Args:
        mask:           Binary mask (uint8, 0/255) to extract the contour from.
        epsilon_factor: Approximation accuracy as a fraction of the contour arc length.
                        Smaller values preserve more detail. Defaults to 0.001.

    Returns:
        List of [x, y] coordinate pairs forming the simplified contour polygon,
        or an empty list if no contours are found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def hair_detection(mask_coordinate: list, left: int) -> float:
    """
    Estimate the shoulder visibility ratio by comparing detected shoulder width
    to the effective image width.

    Identifies the leftmost and rightmost points in the mask contour as the
    left and right shoulder tips respectively, then divides their horizontal
    span by the usable image width (640 px minus the left crop offset on both sides).

    Args:
        mask_coordinate: List of [x, y] contour coordinates from ``mask_to_coords``.
        left:            Horizontal crop offset applied during resizing, used to
                         compute the usable image width.

    Returns:
        float: Shoulder-width-to-image-width ratio. Values close to 1.0 indicate
               both shoulders are fully visible; low values suggest occlusion.
    """
    left_shoulder  = min(mask_coordinate, key=lambda p: (p[0], -p[1]))
    right_shoulder = max(mask_coordinate, key=lambda p: (p[0], -p[1]))
    shoulder_width = right_shoulder[0] - left_shoulder[0]
    width          = 640 - (2 * left)
    width_ratio    = shoulder_width / width
    return width_ratio


def to_json(obj):
    """
    Recursively convert NumPy scalars, arrays, and nested Python containers to
    JSON-serialisable Python types.

    Args:
        obj: Any value to convert. Supported types:
               - None           → None
               - np.ndarray     → list (via ``.tolist()``)
               - np.integer     → int
               - np.floating    → float
               - list or tuple  → list with each element recursively converted
               - Any other type → returned as-is.

    Returns:
        A JSON-serialisable equivalent of ``obj``.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [to_json(i) for i in obj]
    return obj