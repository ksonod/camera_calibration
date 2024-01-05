import numpy as np
from typing import List, Tuple


class CameraCalib:
    checker_shape: Tuple
    checker_size: float
    get_skewness: bool
    num_img_data: int
    img_file_list: List
    points2d: List
    points3d: np.ndarray

    def __init__(self, config: dict, img_file_list: list):
        self.checker_shape = config["checkerboard"]["num_corners"]
        self.checker_size = config["checkerboard"]["checker_size"]
        self.num_img_data = len(img_file_list)
        self.img_file_list = img_file_list
        self.points2d = []  # Detected checkerboard corners in the 2D image plane
        self.points3d = self.create_3d_point_of_checker_corners()  # Corresponding 3D points in space
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
    def calculate_reprojection_error(reference_points2d: np.ndarray, projected_points2d: np.ndarray):
        err = np.squeeze(reference_points2d) - np.squeeze(projected_points2d)
        return np.mean(
                    np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2)
               )
