from pathlib import Path
from algorithm.opencv.calibration import calibrate_with_opencv
from algorithm.matlab.calibration import calibrate_with_matlab
from algorithm.zhang2000.calibration import calibrate_with_zhang_method
from algorithm.general.calib import CalibMethod


INPUT_FILES = {
    "img_folder": Path("./data")
}

CONFIG = {
    "input_file_format": ".jpg",
    # "calibration_method": CalibMethod.ZHANG2000,
    "calibration_method": CalibMethod.OPENCV,
    # "calibration_method": CalibMethod.MATLAB,
    "checkerboard": {
        "num_corners": (8, 5),  # ([numbers of corners per column], [number of corners per row])
        "checker_size": 25,  # mm (millimeter)
        "show_figure": True,
    },
    # "zhang2000": {  # Config for the CalibMethod.ZHANG2000 method.
    #     "get_skewness": False,  # gamma in an intrinsic matrix [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
    #     "optimize_parameters": True
    # }
}


def main(input_files: dict, config: dict):

    if config["calibration_method"] == CalibMethod.OPENCV:
        calibrate_with_opencv(input_files, config)
    elif config["calibration_method"] == CalibMethod.MATLAB:
        calibrate_with_matlab(input_files, config)
    elif config["calibration_method"] == CalibMethod.ZHANG2000:
        calibrate_with_zhang_method(input_files, config)
    else:
        raise NotImplementedError("Select a right calibration method.")


if __name__ == "__main__":
    main(input_files=INPUT_FILES, config=CONFIG)
