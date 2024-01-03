import cv2 as cv
import numpy as np
from typing import Tuple


def create_3d_point_of_checker_corners(checker_shape: Tuple, checker_size: float) -> np.ndarray:

    x = np.arange(0, checker_shape[1], 1)
    y = np.arange(checker_shape[0], 0, -1) - 1

    corners3d = np.stack(
        [
            np.tile(y, checker_shape[1]),
            np.repeat(x, checker_shape[0])
        ], axis=1
    ) * checker_size

    return corners3d.astype(np.float32)


def detect_corners(input_gray_img: np.ndarray, checker_shape: Tuple) -> Tuple[np.ndarray, np.ndarray]:
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
        return np.squeeze(refined_corners)
    else:
        return None
def define_XYZ_coordinate_system(
        rvec: np.ndarray, tvec: np.ndarray, intrinsicK: np.ndarray, distortion_coeff: np.ndarray
):
    """
    Get (X, Y. Z) = (0, 0, 0) and unit vectors along X and Y axes. These values can be used for visualization of X and
    Y axes.
    :param rvec: Rotation vector
    :param tvec: Translation vector
    :param intrinsicK: Camera intrinsics matrix
    :param distortion_coeff: Distortion coefficients. It follows OpenCV convention: k1, k2, p1, p2, k3, ....
    :return:
        - origion_point: (X,Y,Z) = (0, 0, 0)
        - x0: Unit vector in the X direction.
        - y0: Unit vector in the Y direction.
    """

    # Origin point (X, Y, Z) = (0, 0, 0)
    origin_point = cv.projectPoints(
        objectPoints=np.array([0.0, 0.0, 0.0]), rvec=rvec,
        tvec=tvec, cameraMatrix=intrinsicK, distCoeffs=distortion_coeff
    )[0][0][0]

    # Set a unit vector in X
    x0 = cv.projectPoints(
        objectPoints=np.array([1.0, 0.0, 0.0]), rvec=rvec,
        tvec=tvec, cameraMatrix=intrinsicK, distCoeffs=distortion_coeff
    )[0][0][0]

    # Set a unit vector in Y
    y0 = cv.projectPoints(
        objectPoints=np.array([0.0, 1.0, 0.0]), rvec=rvec,
        tvec=tvec, cameraMatrix=intrinsicK, distCoeffs=distortion_coeff
    )[0][0][0]

    return origin_point, x0, y0
