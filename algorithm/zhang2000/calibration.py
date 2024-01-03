import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from algorithm.general.feature_analysis import detect_corners, define_XYZ_coordinate_system
from algorithm.zhang2000.math import vij
from visualization.checkerboard import show_cb_image_with_detected_corners, draw_XY_arrows


def calibrate_with_zhang_method(config: dict, img_file_list: list):
    """
    Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine
    Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.

    Zhang's calibration method is implemented by referring to the aforementioned paper.

    :param config: config dictionary
    :param img_file_list: a list of input image files
    """
    num_img_data = len(img_file_list)

    corners3d = np.zeros(
        (np.prod(config["checkerboard"]["num_corners"]), 3)
    ).astype(np.float32)  # Object points in 3D
    corners3d[:, :2] = np.mgrid[
                            0:config["checkerboard"]["num_corners"][0],
                            0:config["checkerboard"]["num_corners"][1]
                       ].T.reshape(-1, 2) * config["checkerboard"]["checker_size"]  # Z values are always 0.

    V = np.zeros((2 * num_img_data, 6))  # V matrix in Eq. 9
    H = np.zeros((3 * num_img_data, 3))  # Homography matrix

    points2d = []  # 2D points in image plane.
    points3d = []  # 3D points in space

    for idx, img_file in enumerate(img_file_list):
        img = np.array(Image.open(img_file))
        if idx == 0:
            print("Image shape:", img.shape)
        detected_corners2d = detect_corners(
            cv.cvtColor(img, cv.COLOR_RGB2GRAY),
            config["checkerboard"]["num_corners"]
        )

        points2d.append(detected_corners2d)
        points3d.append(corners3d)

        h, _ = cv.findHomography(srcPoints=corners3d, dstPoints=detected_corners2d)

        H[3 * idx:3 * (idx + 1), :] = h

        V[2 * idx, :] = vij(h, 0, 1).T
        V[(2 * idx + 1), :] = (vij(h, 0, 0) - vij(h, 1, 1)).T

    # This commented-out block is described in the paper. Alternatively, using SVD is also suggested.
    # eig_val, eig_vec = np.linalg.eig(V.T @ V)
    # b = eig_vec[:, np.argmin(eig_val)]  # Eq.6

    u, s, v = np.linalg.svd(V)
    b = v[np.argmin(s), :]

    # Equations in the Section 3.1
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
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    np.set_printoptions(precision=3, suppress=True)
    print(f"- Intrinsic parameters : \n{A}")
    print("- Extrinsic parameters")

    # Arrow setting for visualization
    magnification_factor = 30
    head_width = 15
    head_length = 10

    reprojection_error = []

    rvec_list = []
    tvec_list = []
    projected_points2d_list = []

    for i in range(num_img_data):
        h = H[3 * i:3 * (i + 1), :]
        A_inv = np.linalg.inv(A)
        lm = 1 / np.linalg.norm(A_inv @ h[:, 0], ord=2)
        r1 = lm * A_inv @ h[:, 0]
        r2 = lm * A_inv @ h[:, 1]
        r3 = np.cross(r1, r2)
        tvec = lm * A_inv @ h[:, 2]

        R = np.stack([r1, r2, r3]).T
        Rt = np.concatenate([R, tvec.reshape(3, 1)], axis=1)
        rvec = cv.Rodrigues(R)[0].flatten()
        projected_points2d, _ = cv.projectPoints(
            objectPoints=points3d[i], rvec=rvec, tvec=tvec, cameraMatrix=A, distCoeffs=None
        )
        projected_points2d_list.append(projected_points2d)
        rvec_list.append(rvec)
        tvec_list.append(tvec)
        err = np.squeeze(projected_points2d) - points2d[i]

        # averaged over all the corners detected in a single image
        reprojection_error.append(
            np.mean(np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2))
        )

        print(f"{img_file_list[i].name} | Reprojection error = {reprojection_error[-1]:.5f}")
        print(f"[R | t]:\n{Rt}")
        print("Rot vec: ", rvec, "\n")

        if config["checkerboard"]["show_figure"]:
            plt.figure()
            img = np.array(Image.open(img_file_list[i]))

            show_cb_image_with_detected_corners(
                img=img, detected_points=points2d[i], figure_title=f"{img_file_list[i].name}"
            )

            # Set an origin (X, Y, Z) = (0, 0, 0) and unit vectors in X and Y directions.
            origin_point, x0, y0 = define_XYZ_coordinate_system(
                rvec=rvec, tvec=tvec, intrinsicK=A, distortion_coeff=None
            )

            draw_XY_arrows(
                origin_point=origin_point,
                x0=x0,
                y0=y0,
                magnification_factor=magnification_factor,
                head_width=head_width,
                head_length=head_length,
            )

    print("- Mean reprojection error")
    print(f" Overall: {np.mean(reprojection_error):.5f}")  # Averaging reprojection errors over all images

    plt.figure()
    plt.bar(
        np.arange(num_img_data), reprojection_error, color="blue", alpha=0.5
    )
    plt.plot(
        [-0.5, num_img_data - 0.5], np.mean(reprojection_error) * np.ones(2), "k--"
    )
    plt.xlabel("Images")
    plt.ylabel("Mean reprojection error (pixel)")
    plt.show()
