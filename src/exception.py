class DetectionFailed(Exception):
    """
    Raised when a detection model fails to produce a valid result.

    Typically caught by the FastAPI exception handler and returned to the
    client as a 422 JSON response with error_code ``DETECTION_FAILED``.

    Attributes:
        method      (str | None): Name of the detection step that failed
                                  (e.g. ``"Neck Detection"``).
        status_code (int)       : HTTP status code to return to the client.
                                  Defaults to 422.
    """

    def __init__(self, status_code: int = 422, method: str = None, cause: str = None) -> None:
        """
        Args:
            status_code: HTTP status code for the error response. Defaults to 422.
            method:      Name of the detection method that failed.
                         Included in the exception message.
            cause:       Optional detail string describing the root cause.
                         Appended to the message as `` | <cause>`` when provided.
        """
        self.method      = method
        self.status_code = status_code
        detail = f" | {cause}" if cause else ""
        super().__init__(f"{method} Failed{detail}")


class ModelLoadError(Exception):
    """
    Raised when a model or classifier cannot be loaded at startup.

    Typically caught by the FastAPI exception handler and returned to the
    client as a 500 JSON response with error_code ``MODEL_LOAD_ERROR``.

    Attributes:
        method      (str | None): Name of the model or loader that failed
                                  (e.g. ``"Bbox ONNX"``, ``"Haar Cascade"``).
        status_code (int)       : HTTP status code to return to the client.
                                  Defaults to 500.
    """

    def __init__(self, status_code: int = 500, method: str = None, cause: str = None) -> None:
        """
        Args:
            status_code: HTTP status code for the error response. Defaults to 500.
            method:      Name of the model or loading step that failed.
                         Included in the exception message as ``"Load <method> Failed"``.
            cause:       Optional detail string describing the root cause.
                         Appended to the message as `` | <cause>`` when provided.
        """
        self.method      = method
        self.status_code = status_code
        detail = f" | {cause}" if cause else ""
        super().__init__(f"Load {method} Failed{detail}")


class ShoulderNeckNotVisible(Exception):
    """
    Raised when the shoulder or neck region is not sufficiently visible in the image.

    Used to signal that the input image does not meet the geometric requirements
    for keypoint detection or mask extraction (e.g. the person is too far away,
    heavily occluded, or the camera angle makes the neck region undetectable).

    Attributes:
        status_code (int): HTTP status code to return to the client. Always 422.
    """

    def __init__(self, cause: str = "") -> None:
        """
        Args:
            cause: Human-readable description of why the shoulder/neck is not
                   visible. Passed directly as the exception message.
        """
        self.status_code = 422
        super().__init__(cause)