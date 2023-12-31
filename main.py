import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple

INPUT_FILES = {
    "img_folder": Path("./data")
}


CONFIG = {
    "checkerboard": {
        "num_corners": (9, 6),  # ([numbers of corners per column], [number of corners per row])
        "checker_size": 21.5,  # mm
        "show_figure": False,
    }
}


def detect_corners(
        input_gray_img: np.ndarray, checker_shape: Tuple, show_figure: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detected, corners = cv2.findChessboardCorners(
        input_gray_img,
        checker_shape,
        None
    )

    if detected:
        refined_corners = cv2.cornerSubPix(input_gray_img, corners, (11, 11), (-1, -1), criteria)

        if show_figure:
            cv2.drawChessboardCorners(
                input_gray_img,
                checker_shape,
                refined_corners,
                detected
            )
            cv2.imshow("img", input_gray_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return np.squeeze(refined_corners)
        # return np.squeeze(refined_corners)[:, [1,0]]


def vij(h: np.ndarray, i: int, j: int):
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
    # return np.array(
    #     [
    #         h[i, 0] * h[j, 0],
    #         h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
    #         h[i, 1] * h[j, 1],
    #         h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2],
    #         h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2],
    #         h[i, 2] * h[j, 2]
    #
    #     ]
    # ).reshape(6, 1)


def run_scripts(input_files: dict, config: dict):

    x = np.arange(0, config["checkerboard"]["num_corners"][1], 1)
    y = np.arange(config["checkerboard"]["num_corners"][0], 0, -1) - 1

    corners3d = np.stack(
        [
            np.tile(y, config["checkerboard"]["num_corners"][1]),
            np.repeat(x, config["checkerboard"]["num_corners"][0])
        ], axis=1
    ) * config["checkerboard"]["checker_size"]

    # corners3d = np.stack(
    #     [
    #         np.repeat(x, config["checkerboard"]["num_corners"][0]),
    #         np.tile(y, config["checkerboard"]["num_corners"][1])
    #     ], axis=1
    # ) * config["checkerboard"]["checker_size"]


    img_file_list = list(input_files["img_folder"].glob("*.jpg"))
    img_file_list.sort()
    num_img_data = len(img_file_list)

    V = np.zeros((2*num_img_data, 6))  # V matrix in Eq. 9
    H = np.zeros((3*num_img_data, 3))

    for idx, img_file in enumerate(img_file_list):
        img = np.array(Image.open(img_file))
        if idx == 0:
            print("Image shape:", img.shape)
        corners2d = detect_corners(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            config["checkerboard"]["num_corners"],
            config["checkerboard"]["show_figure"]
        )
        h, _ = cv2.findHomography(srcPoints=corners3d, dstPoints=corners2d)

        H[3*idx:3*(idx+1), :] = h

        V[2*idx, :] = vij(h, 0, 1).T
        V[(2*idx+1), :] = (vij(h, 0, 0) - vij(h, 1, 1)).T

    eig_val, eig_vec = np.linalg.eig(V.T @ V)
    b = eig_vec[:, np.argmin(eig_val)]  # Eq.6
    # u, s, v = np.linalg.svd(V)
    # b = v[np.argmin(s), :]
    # print(b)

    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = np.sqrt(l / b[0])
    beta = np.sqrt(l * b[0] / (b[0] * b[2] - b[1]**2))
    gamma = - b[1] * alpha**2 * beta / l
    u0 = gamma*v0/alpha - b[3]*alpha**2/l
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    for i in range(num_img_data):
        h = H[3*i:3*(i+1), :]
        A_inv = np.linalg.inv(A)
        lm = 1/np.linalg.norm(A_inv @ h[:, 0], ord=2)
        r1 = lm * A_inv @ h[:, 0]
        r2 = lm * A_inv @ h[:, 1]
        r3 = np.cross(r1, r2)
        t = lm * A_inv @ h[:, 2]

        R = np.stack([r1, r2, r3]).T
        print(R)
        print(t)

    print(A)
    return 0  # TODO: remove it


if __name__ == "__main__":
    run_scripts(input_files=INPUT_FILES, config=CONFIG)
