from algorithm.zhang2000.zhang2000calib import Zhang2000Calib


def calibrate_with_zhang_method(config: dict, img_file_list: list):
    """
    Zhang's calibration method is implemented by referring to this paper:
    Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine
    Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.

    :param config: config dictionary
    :param img_file_list: a list of input image files
    """

    calib = Zhang2000Calib(config, img_file_list)
    calib()
