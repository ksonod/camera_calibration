import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from algorithm.general.feature_analysis import detect_corners, create_3d_point_of_checker_corners
from algorithm.zhang2000.math import vij

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


def run_scripts(input_files: dict, config: dict):
    corners3d = create_3d_point_of_checker_corners(
        checker_shape=config["checkerboard"]["num_corners"],
        checker_size=config["checkerboard"]["checker_size"]
    )

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

    # Equations in the Section 3.1
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = np.sqrt(l / b[0])
    beta = np.sqrt(l * b[0] / (b[0] * b[2] - b[1]**2))
    gamma = - b[1] * alpha**2 * beta / l
    u0 = gamma*v0/alpha - b[3]*alpha**2/l

    # Eq. 1
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    print(f"- Intrinsics Matrix - \n{A}\n")

    for i in range(num_img_data):
        h = H[3*i:3*(i+1), :]
        A_inv = np.linalg.inv(A)
        lm = 1/np.linalg.norm(A_inv @ h[:, 0], ord=2)
        r1 = lm * A_inv @ h[:, 0]
        r2 = lm * A_inv @ h[:, 1]
        r3 = np.cross(r1, r2)
        t = lm * A_inv @ h[:, 2]

        R = np.stack([r1, r2, r3]).T
        Rt = np.concatenate([R, t.reshape(3,1)], axis=1)
        print(f"[R | t]:\n{Rt}\n")


if __name__ == "__main__":
    run_scripts(input_files=INPUT_FILES, config=CONFIG)
