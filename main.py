from pathlib import Path
from enum import Enum
from algorithm.opencv.calibration import calibrate_with_opencv
from algorithm.matlab.calibration import calibrate_with_matlab
from algorithm.zhang2000.calibration import calibrate_with_zhang_method


class CalibMethod(Enum):
    OPENCV = "opencv"
    MATLAB = "matlab"
    ZHANG2000 = "zhang2000"


INPUT_FILES = {
    "img_folder": Path("./data")
}

CONFIG = {
    "input_file_format": ".jpg",
    "calibration_method": CalibMethod.ZHANG2000,
    # "calibration_method": CalibMethod.OPENCV,
    # "calibration_method": CalibMethod.MATLAB,
    "checkerboard": {
        "num_corners": (8, 5),  # ([numbers of corners per column], [number of corners per row])
        "checker_size": 25,  # mm (millimeter)
        "show_figure": True,
    },
    "zhang2000": {
        "get_skewness": False,  # gamma in an intrinsic matrix [[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]
        "optimize_parameters": True
    }
}


def run_scripts(input_files: dict, config: dict):

    img_file_list = list(input_files["img_folder"].glob(f"*{config['input_file_format']}"))
    img_file_list.sort()

    print(f"Calibration method: {config['calibration_method'].value}")

    if config["calibration_method"] == CalibMethod.OPENCV:
        calibrate_with_opencv(config, img_file_list)
    elif config["calibration_method"] == CalibMethod.MATLAB:
        calibrate_with_matlab(config, img_file_list)
    elif config["calibration_method"] == CalibMethod.ZHANG2000:
        calibrate_with_zhang_method(config, img_file_list)
    else:
        raise NotImplementedError("Select a right calibration method.")


if __name__ == "__main__":
    run_scripts(input_files=INPUT_FILES, config=CONFIG)
