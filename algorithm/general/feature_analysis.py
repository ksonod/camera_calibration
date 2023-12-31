import cv2 as cv
import numpy as np
from typing import Tuple


def create_3d_point_of_checker_corners(checker_shape: Tuple, checker_size: float) -> np.ndarray:
    """
    Order of points is aligned with OpenCV checker detection.

    """

    x = np.arange(0, checker_shape[1], 1)
    y = np.arange(checker_shape[0], 0, -1) - 1

    corners3d = np.stack(
        [
            np.tile(y, checker_shape[1]),
            np.repeat(x, checker_shape[0])
        ], axis=1
    ) * checker_size

    return corners3d


def detect_corners(
        input_gray_img: np.ndarray, checker_shape: Tuple, show_figure: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detected, corners = cv.findChessboardCorners(
        input_gray_img,
        checker_shape,
        None
    )

    if detected:
        refined_corners = cv.cornerSubPix(input_gray_img, corners, (11, 11), (-1, -1), criteria)

        if show_figure:
            cv.drawChessboardCorners(
                input_gray_img,
                checker_shape,
                refined_corners,
                detected
            )
            cv.imshow("img", input_gray_img)
            cv.waitKey()
            cv.destroyAllWindows()
        return np.squeeze(refined_corners)
        # return np.squeeze(refined_corners)[:, [1,0]]
