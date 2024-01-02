import numpy as np


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
