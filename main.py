from pathlib import Path
from enum import Enum
from algorithm.zhang2000.calibration import calibrate_with_zhang_method
from algorithm.opencv.calibration import calibrate_with_opencv
from algorithm.matlab.calibration import calibrate_with_matlab


class CalibMethod(Enum):
    ZHANG2000 = "zhang2000"
    OPENCV = "opencv"
    MATLAB = "matlab"


INPUT_FILES = {
    "img_folder": Path("./data")
}

CONFIG = {
    "input_file_format": ".jpg",
    "calibration_method": CalibMethod.MATLAB,
    "checkerboard": {
        "num_corners": (9, 6),  # ([numbers of corners per column], [number of corners per row])
        "checker_size": 21.5,  # mm (millimeter)
        "show_figure": False,
    },
}


def run_scripts(input_files: dict, config: dict):

    img_file_list = list(input_files["img_folder"].glob(f"*{config['input_file_format']}"))
    img_file_list.sort()

    print(f"Calibration method: {config['calibration_method'].value}")

    if config["calibration_method"] == CalibMethod.ZHANG2000:
        calibrate_with_zhang_method(config, img_file_list)
    elif config["calibration_method"] == CalibMethod.OPENCV:
        calibrate_with_opencv(config, img_file_list)
    elif config["calibration_method"] == CalibMethod.MATLAB:
        calibrate_with_matlab(config, img_file_list)
    else:
        raise NotImplementedError("Select a right calibration method.")


if __name__ == "__main__":
    run_scripts(input_files=INPUT_FILES, config=CONFIG)
