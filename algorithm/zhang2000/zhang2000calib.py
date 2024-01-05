import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import least_squares
from typing import Tuple, List
from algorithm.general.feature_analysis import detect_corners, define_XYZ_coordinate_system
from visualization.checkerboard import show_cb_image_with_detected_corners, draw_XY_arrows
from algorithm.general.calib import CameraCalib


class Zhang2000Calib(CameraCalib):
    """
    [1] Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine
    Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.
    [2] Z. Zhang (published on Dec. 2, 1998, updated on Aug. 13, 2008) "A Flexible New Technique for Camera
    Calibration." microsoft.com [Online]. Available:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf [Accessed: Jan. 5, 2024]

    Zhang's calibration method is implemented by referring to the aforementioned paper.

    :param config: config dictionary
    :param img_file_list: a list of input image files
    """

    def __init__(self, config: dict, img_file_list: list):
        super().__init__(config, img_file_list)

    def __call__(self):
        V, H = self.get_V_and_H()  # V matrix from Eq (9) in [1] and homography matrix H
        b = self.get_b_vector(V)  # b vector in Eq. 6 in [1]
        A = self.get_intrinsic_params(b)  # Intrinsic parameters
        alpha, beta, gamma, u0, v0 = A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2]  # Eq 1 in [1]
        np.set_printoptions(precision=3, suppress=True)
        print(f"- Intrinsic parameters : \n{A}")
        print("- Extrinsic parameters")

        # Arrow setting for visualization
        magnification_factor = 30
        head_width = 15
        head_length = 10

        # List for storing extrinsic parameters
        rvec_list = []  # Rotation vector
        tvec_list = []  # Translation vector

        reprojection_error = []  # Reprojection error
        projected_points2d_original_list = []  # Projected points using the first non-optimized camera parameters

        # Initialization of D matrix and d vector from Eq. 13 in [2]
        D = np.zeros(
            (2 * np.prod(self.checker_shape) * self.num_img_data, 2)
        )
        d = np.zeros(
            (2 * np.prod(self.checker_shape) * self.num_img_data, 1)
        )

        for i in range(self.num_img_data):
            Rt, rvec, tvec = self.get_rvec_and_tvec(H=H, A=A, idx=i)
            projected_points2d, _ = cv.projectPoints(
                objectPoints=self.points3d,
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=A,
                distCoeffs=np.array([0, 0, 0, 0]).astype(np.float32)
            )
            projected_points2d_original_list.append(projected_points2d)
            rvec_list.append(rvec)
            tvec_list.append(tvec)

            err = np.squeeze(projected_points2d) - self.points2d[i]

            # averaged over all the corners detected in a single image
            reprojection_error.append(
                np.mean(np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2))
            )

            for j, point in enumerate(np.squeeze(projected_points2d)):
                x, y = (point - [u0, v0]) / [alpha, beta]

                r = x ** 2 + y ** 2

                D[2 * i * np.prod(self.checker_shape) + 2 * j, :] = [
                    (point[0] - u0) * r,
                    (point[0] - u0) * r ** 2
                ]
                D[2 * i * np.prod(self.checker_shape) + 2 * j + 1, :] = [
                    (point[1] - v0) * r,
                    (point[1] - v0) * r ** 2
                ]

                point_idx = np.argmin(np.prod(np.abs(self.points2d[i] - point), axis=1))
                d[2 * i * np.prod(self.checker_shape) + 2 * j, 0] = self.points2d[i][point_idx][0] - point[0]
                d[2 * i * np.prod(self.checker_shape) + 2 * j + 1, 0] = self.points2d[i][point_idx][1] - point[1]

            print(f"{self.img_file_list[i].name} | Reprojection error = {reprojection_error[-1]:.5f}")
            print(f"[R | t]:\n{Rt}")
            print("Rot vec: ", rvec, "\n")

        k = np.linalg.inv(D.T @ D) @ D.T @ d  # Radial distortion parameters (Eq. 13 in [2])
        k1, k2 = k.flatten()
        print(f" Initial distortion params: {k1}, {k2}\n")

        print("- Mean reprojection error")
        print(f" Overall: {np.mean(reprojection_error):.5f}")  # Averaging reprojection errors over all images

        if self.get_skewness:  # With gamma
            param_scale = [alpha, beta, gamma, u0, v0, k1, k2]  # alpha, beta, gamma, u0, v0, k1, k2
        else:  # Without gamma
            param_scale = [alpha, beta, u0, v0, k1, k2]  # alpha, beta, u0, v0, k1, k2
        num_intrinsic_params = len(param_scale)

        for rvec_value in np.array(rvec_list).flatten():  # Add rotational vectors as parameters
            param_scale.append(rvec_value)
        for tvec_value in np.array(tvec_list).flatten():  # Add translation vectors as parameters
            param_scale.append(tvec_value)

        param_scale = np.array(param_scale).astype(np.float32)
        initial_params = np.ones_like(param_scale).astype(np.float32)  # Normalized array to be optimized.

        optimized_params = least_squares(
            fun=self.loss,
            x0=initial_params,
            args=(self.points2d, self.get_skewness, param_scale)
        )

        updated_params = optimized_params.x * param_scale

        # Get a new intrinsics matrix.
        if self.get_skewness:
            A_new = np.array([
                [updated_params[0], updated_params[2], updated_params[3]],  # With gamma
                [0, updated_params[1], updated_params[4]],
                [0, 0, 1]
            ])
        else:
            A_new = np.array([
                [updated_params[0], 0, updated_params[2]],  # Without gamma
                [0, updated_params[1], updated_params[3]],
                [0, 0, 1]
            ]).astype(np.float32)

        # Get radial distortion parameters k1 and k2
        if self.get_skewness:
            distortion_coef = np.array([updated_params[5], updated_params[6], 0, 0])
        else:
            distortion_coef = np.array([updated_params[4], updated_params[5], 0, 0])

        # Calculate reprojection errors using the new camera parameters.
        reprojection_error_new = []
        for i in range(self.num_img_data):
            img = np.array(Image.open(self.img_file_list[i]))

            rvec_new = updated_params[
                            (num_intrinsic_params + 3 * i):(num_intrinsic_params + 3 * (i + 1))
                       ]
            tvec_new = updated_params[
                            (num_intrinsic_params + 3 * self.num_img_data + (3 * i)):(
                             num_intrinsic_params + 3 * self.num_img_data + 3 * (i + 1))
                       ]

            new_projected_points2d, _ = cv.projectPoints(
                objectPoints=self.points3d,
                rvec=rvec_new,
                tvec=tvec_new,
                cameraMatrix=A_new,
                distCoeffs=distortion_coef
            )

            err = np.squeeze(new_projected_points2d) - np.squeeze(self.points2d[i])
            reprojection_error_new.append(np.mean(np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2)))

            if self.show_figure:

                plt.figure()

                show_cb_image_with_detected_corners(
                    img=img, detected_points=self.points2d[i], figure_title=f"{self.img_file_list[i].name}"
                )

                for idx_points in range(new_projected_points2d.shape[0]):
                    plt.plot(new_projected_points2d[idx_points, 0, 0], new_projected_points2d[idx_points, 0, 1], "r.")
                    plt.plot(projected_points2d_original_list[i][idx_points, 0, 0],
                             projected_points2d_original_list[i][idx_points, 0, 1], marker="x", color="pink")

                # Set an origin (X, Y, Z) = (0, 0, 0) and unit vectors in X and Y directions.
                origin_point, x0, y0 = define_XYZ_coordinate_system(
                    rvec=rvec_list[i], tvec=tvec_list[i], intrinsicK=A, distortion_coeff=np.array([0.0, 0.0, 0.0, 0.0])
                )

                draw_XY_arrows(
                    origin_point=origin_point,
                    x0=x0,
                    y0=y0,
                    magnification_factor=magnification_factor,
                    head_width=head_width,
                    head_length=head_length,
                )

        print(" After optimization", np.mean(reprojection_error_new))

        # Show final results (reprojection errors)
        plt.figure()
        plt.bar(
            np.arange(self.num_img_data), reprojection_error, color="blue", alpha=0.5
        )

        plt.bar(
            np.arange(self.num_img_data)+0.1, reprojection_error_new, color="red", alpha=0.5
        )

        plt.plot(
            [-0.6, self.num_img_data - 0.3], np.mean(reprojection_error) * np.ones(2), "b--", alpha=0.5
        )
        plt.plot(
            [-0.6, self.num_img_data - 0.3], np.mean(reprojection_error_new) * np.ones(2), "r--", alpha=0.5
        )

        plt.xlabel("Images")
        plt.ylabel("Mean reprojection error (pixel)")
        plt.xlim([-0.6, self.num_img_data-0.3])
        plt.show()

    @staticmethod
    def vij(h: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        :param h: Homography matrix defined in Eq. 2.
        :param i: matrix index 1 (column number of H matrix)
        :param j: matrix index 2 (column number of H matrix)
        :return: v vector defined in Eq. 7.
        """
        return np.array(
            [
                h[0, i] * h[0, j],
                h[0, i] * h[1, j] + h[1, i] * h[0, j],
                h[1, i] * h[1, j],
                h[2, i] * h[0, j] + h[0, i] * h[2, j],
                h[2, i] * h[1, j] + h[1, i] * h[2, j],
                h[2, i] * h[2, j]

            ]
        ).reshape(6, 1)

    def get_V_and_H(self) -> Tuple[np.ndarray, np.ndarray]:
        H = np.zeros((3 * self.num_img_data, 3))  # Homography matrix
        V = np.zeros(
            (2 * self.num_img_data, 6))  # V matrix in Eq. 9, which includes some elements from the homography matrix

        for idx, img_file in enumerate(self.img_file_list):

            img = np.array(Image.open(img_file))

            if idx == 0:
                print("Image shape:", img.shape)

            detected_corners2d = detect_corners(
                cv.cvtColor(img, cv.COLOR_RGB2GRAY),
                self.checker_shape
            )

            self.points2d.append(detected_corners2d)

            h, _ = cv.findHomography(srcPoints=self.points3d, dstPoints=detected_corners2d)

            H[3 * idx:3 * (idx + 1), :] = h

            V[2 * idx, :] = self.vij(h, 0, 1).T
            V[(2 * idx + 1), :] = (self.vij(h, 0, 0) - self.vij(h, 1, 1)).T

        if not self.get_skewness:
            V = V[:, [0, 2, 3, 4, 5]]
        return V, H

    def get_b_vector(self, V: np.ndarray) -> np.ndarray:
        u, s, v = np.linalg.svd(V)
        b = v[np.argmin(s), :]  # b = [B11, B12, B22, B13, B23, B33]

        # This commented-out block is described in the paper. Alternatively, using SVD is also suggested.
        # eig_val, eig_vec = np.linalg.eig(V.T @ V)
        # b = eig_vec[:, np.argmin(eig_val)]  # Eq.6

        if not self.get_skewness:
            b = np.insert(arr=b, obj=1, values=0)  # B11, B12=0, B22, B13, B23, B33

        return b

    @staticmethod
    def get_intrinsic_params(b: np.ndarray) -> np.ndarray:
        """
        Equations in the Section 3.1 are used to calculate intrinsic parameters from the 6D vector b.

        :param b: 6D vector b is defined in Eq. 6
        :return:
        """
        # (B_12 * B_13 - B_11 * B_23) / (B_11 * B22 - B_12 ** 2)
        v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)

        # lambda = B_33 - [B_13 ** 2 + v0 * (B_12 * B_13 - B_11 * B_23)] / B_11
        l = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]

        # alpha = sqrt(lambda / B_11)
        alpha = np.sqrt(l / b[0])

        # beta = sqrt(lambda * B_11 / (B_11 * B_22 - B_12 ** 2) )
        beta = np.sqrt(l * b[0] / (b[0] * b[2] - b[1] ** 2))

        # gamma = - B_12 * alpha ** 2 * beta / lambda
        gamma = - b[1] * alpha ** 2 * beta / l

        # The equation for u0 is incorrectly written in the original paper:
        #  Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine
        #  Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.
        # A correct formula can be found in:
        # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf [Accessed: Jan. 3, 2024]
        # u0 = gamma * v0 / beta - B_13 * alpha ** 2 / lambda
        u0 = gamma * v0 / beta - b[3] * alpha ** 2 / l

        # Eq. 1
        return np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ]).astype(np.float32)

    @staticmethod
    def get_rvec_and_tvec(
            H: np.ndarray, A: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = H[3 * idx:3 * (idx + 1), :]
        A_inv = np.linalg.inv(A)
        lm = 1 / np.linalg.norm(A_inv @ h[:, 0], ord=2)
        r1 = lm * A_inv @ h[:, 0]
        r2 = lm * A_inv @ h[:, 1]
        r3 = np.cross(r1, r2)
        tvec = lm * A_inv @ h[:, 2]

        R = np.stack([r1, r2, r3]).T
        Rt = np.concatenate([R, tvec.reshape(3, 1)], axis=1)
        rvec = cv.Rodrigues(R)[0].flatten()
        return Rt, rvec, tvec

    def loss(self, params: List, points2d: List, gamma_available: bool, param_scale: np.ndarray) -> np.ndarray:
        params = np.array(params) * param_scale

        if gamma_available:
            alpha, beta, gamma, u0, v0, k1, k2 = params[:7]
            num_intrinsic_params = 7
        else:
            alpha, beta, u0, v0, k1, k2 = params[:6]
            gamma = 0
            num_intrinsic_params = 6

        num_image_data = int(len(params[num_intrinsic_params:])/6)

        total_error = 0

        for i in range(num_image_data):
            projected_points2d, _ = cv.projectPoints(
                objectPoints=self.points3d,
                rvec=params[
                        (num_intrinsic_params + 3 * i):(
                         num_intrinsic_params + 3 * (i + 1)
                        )
                     ],
                tvec=params[
                        (num_intrinsic_params + 3 * num_image_data + (3 * i)):(
                         num_intrinsic_params + 3 * num_image_data + 3 * (i + 1))
                     ],
                cameraMatrix=np.array(
                    [
                        [alpha, gamma, u0],
                        [0, beta, v0],
                        [0, 0, 1]
                    ]
                ).astype(np.float32),
                distCoeffs=np.array([k1, k2, 0, 0]).astype(np.float32)
            )

            projected_points2d = np.squeeze(projected_points2d)
            err = projected_points2d - np.squeeze(points2d[i])
            total_error += np.sum(err[:, 0] ** 2 + err[:, 1] ** 2)   # Mean and root are not necessary.

        # print(f"error: {total_error} | params: alpha={alpha} - beta={beta} - gamma={gamma} - u0={u0} - v0={v0} - k1={k1} - k2={k2}")
        return total_error.astype(np.float32)
