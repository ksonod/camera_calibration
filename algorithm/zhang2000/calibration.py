from algorithm.zhang2000.zhang2000calib import Zhang2000Calib


def calibrate_with_zhang_method(config: dict, img_file_list: list):
    """
    Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine
    Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.

    Zhang's calibration method is implemented by referring to the aforementioned paper.

    :param config: config dictionary
    :param img_file_list: a list of input image files
    """

    calib = Zhang2000Calib(config, img_file_list)
    calib()
