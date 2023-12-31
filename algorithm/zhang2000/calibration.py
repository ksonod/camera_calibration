import numpy as np
import cv2 as cv
from PIL import Image
from algorithm.general.feature_analysis import detect_corners, create_3d_point_of_checker_corners
from algorithm.zhang2000.math import vij


def calibrate_with_zhang_method(config: dict, img_file_list: list):

    num_img_data = len(img_file_list)

    corners3d = create_3d_point_of_checker_corners(
        checker_shape=config["checkerboard"]["num_corners"],
        checker_size=config["checkerboard"]["checker_size"]
    )

    V = np.zeros((2 * num_img_data, 6))  # V matrix in Eq. 9
    H = np.zeros((3 * num_img_data, 3))  # Homography matrix

    for idx, img_file in enumerate(img_file_list):
        img = np.array(Image.open(img_file))
        if idx == 0:
            print("Image shape:", img.shape)
        corners2d = detect_corners(
            cv.cvtColor(img, cv.COLOR_RGB2GRAY),
            config["checkerboard"]["num_corners"],
            config["checkerboard"]["show_figure"]
        )
        h, _ = cv.findHomography(srcPoints=corners3d, dstPoints=corners2d)

        H[3 * idx:3 * (idx + 1), :] = h

        V[2 * idx, :] = vij(h, 0, 1).T
        V[(2 * idx + 1), :] = (vij(h, 0, 0) - vij(h, 1, 1)).T

    eig_val, eig_vec = np.linalg.eig(V.T @ V)
    b = eig_vec[:, np.argmin(eig_val)]  # Eq.6
    # u, s, v = np.linalg.svd(V)
    # b = v[np.argmin(s), :]

    # Equations in the Section 3.1
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)
    l = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = np.sqrt(l / b[0])
    beta = np.sqrt(l * b[0] / (b[0] * b[2] - b[1] ** 2))
    gamma = - b[1] * alpha ** 2 * beta / l
    u0 = gamma * v0 / alpha - b[3] * alpha ** 2 / l

    # Eq. 1
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    print(f"- Intrinsics Matrix - \n{A}\n")

    for i in range(num_img_data):
        print(img_file_list[i])
        h = H[3 * i:3 * (i + 1), :]
        A_inv = np.linalg.inv(A)
        lm = 1 / np.linalg.norm(A_inv @ h[:, 0], ord=2)
        r1 = lm * A_inv @ h[:, 0]
        r2 = lm * A_inv @ h[:, 1]
        r3 = np.cross(r1, r2)
        t = lm * A_inv @ h[:, 2]

        R = np.stack([r1, r2, r3]).T
        Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
        # print(f"[R | t]:\n{Rt}\n")
        print("Rot vec: ", cv.Rodrigues(R)[0].flatten())
        print("Trans vec: ", t.flatten(), "\n")
