import numpy as np
from typing import List, Tuple
from enum import Enum


class CalibMethod(Enum):
    OPENCV = "opencv"
    MATLAB = "matlab"
    ZHANG2000 = "zhang2000"


class CameraCalib:
    checker_shape: Tuple  # Checkerboard shape: ([numbers of corners per column], [number of corners per row])
    checker_size: float  # Size of one square in a checkerboard in mm.
    get_skewness: bool  # Boolean for getting skewness (gamma) in an intrinsics matrix
    num_img_data: int  # Number of image data
    img_file_list: List  # Image file list
    points2d: List   # Detected checkerboard corners in the 2D image plane
    points3d: np.ndarray   # Corresponding 3D points in space

    def __init__(self, config: dict, img_file_list: list):
        self.num_img_data = len(img_file_list)

        if self.num_img_data == 0:
            raise FileNotFoundError(
                "No image files are found. Check directory name or image data type (jpg, png, and so on)"
            )

        self.img_file_list = img_file_list
        self.checker_shape = config["checkerboard"]["num_corners"]
        self.checker_size = config["checkerboard"]["checker_size"]
        self.points2d = []
        self.points3d = self.create_3d_point_of_checker_corners()
        self.show_figure = config["checkerboard"]["show_figure"]

    def create_3d_point_of_checker_corners(self) -> np.ndarray:

        world_points3d = np.zeros(
            (np.prod(self.checker_shape), 3)
        ).astype(np.float32)  # Object points in 3D
        world_points3d[:, :2] = np.mgrid[
                                0:self.checker_shape[0],
                                0:self.checker_shape[1]
                           ].T.reshape(-1, 2) * self.checker_size  # Z values are always 0.

        # x = np.arange(0, checker_shape[1], 1)
        # y = np.arange(checker_shape[0], 0, -1) - 1
        #
        # corners3d = np.stack(
        #     [
        #         np.tile(y, checker_shape[1]),
        #         np.repeat(x, checker_shape[0])
        #     ], axis=1
        # ) * checker_size

        return world_points3d.astype(np.float32)

    @staticmethod
    def calculate_reprojection_error(reference_points2d: np.ndarray, projected_points2d: np.ndarray) -> float:
        err = np.squeeze(reference_points2d) - np.squeeze(projected_points2d)
        return np.mean(
                    np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2)
               )
