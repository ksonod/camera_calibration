import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path


def calibrate_with_matlab(config: dict, img_file_list: list):
    """
    Tested with MATLAB R2023b + python 3.10
    """
    import matlab.engine
    print("Start MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath("algorithm/matlab")

    num_img_files = len(img_file_list)
    str_img_file_list = eng.cell(1, num_img_files)

    # Convert the file list type for MATLAB
    for idx, path_file in enumerate(img_file_list):
        str_img_file_list[idx] = img_file_list[idx].absolute().__str__()

    # Run matlab script for calibration and save calibration data in .mat files in ./data folder.
    print("Run a MATLAB calibration script...")
    try:
        eng.calib_with_matlab(str_img_file_list, config["checkerboard"]["checker_size"], nargout=0)
    except:
        print("MATLAB did not run successfully.")

    data_folder = Path.cwd() / "data"

    if len(list(data_folder.glob("*.mat"))) != 0:

        print("*.mat files exist. Start loading and processing data.")

        A, K, radial_distortion, tangential_distortion, reprojection_error = load_mat_files(data_folder)
        absolute_reproject_err = np.sqrt(reprojection_error[:, 0, :]**2 + reprojection_error[:, 1, :]**2)

        print("Finished processing data...")

        plt.figure()
        plt.bar(
            np.arange(num_img_files), np.mean(absolute_reproject_err, axis=0), color="blue", alpha=0.5
        )
        plt.plot(
            [-0.5, num_img_files-0.5], np.mean(absolute_reproject_err) * np.ones(2), "k--"
        )
        plt.xlabel("Images")
        plt.ylabel("Mean reprojection error (pixel)")
        plt.show()
    else:
        raise FileNotFoundError("*.mat files do not exist.")

    eng.quit()  # Terminate the MATLAB engine.


def load_mat_files(data_folder):

    # Get extrinsics parameters (3 x 4 matrix [R | t])
    a = loadmat(data_folder / "extrinsicsA.mat")
    A = a['A']

    # Get intrinsics parameters (3 x 3 matrix)
    k = loadmat(data_folder / "intrinsicsK.mat")
    K = k["K"]

    # Get radial distortion parameters
    rd = loadmat(data_folder / "radialDistortion.mat")
    radial_distortion = rd["rd"]

    # Get tangential distortion parameters
    td = loadmat(data_folder / "tangentialDistortion.mat")
    tangential_distortion = td["td"]

    # Get reprojection errors
    re = loadmat(data_folder / "reprojectionError.mat")
    reprojection_error = re["re"]  # [number of detected points in a single image] x 2 x [number of images]

    return A, K, radial_distortion, tangential_distortion, reprojection_error
