import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
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
        absolute_reproject_err = np.sqrt(reprojection_error[:, 0, :] ** 2 + reprojection_error[:, 1, :] ** 2)

        # Reprojection error is averaged over all the detected points for each image
        mean_abs_reproject_err = np.mean(absolute_reproject_err, axis=0)

        print("Finished processing data... \n")

        np.set_printoptions(precision=3, suppress=True)
        print(f"- Intrinsic parameters : \n{K}\n")

        print("- Extrinsic parameters")
        for idx in range(num_img_files):
            print(f"{img_file_list[idx].name} | Reprojection error = {mean_abs_reproject_err[idx]:.5f}")
            print(
                "Rot. vec: ",
                np.array(
                    eng.rotmat2vec3d(
                        matlab.double(A[0, idx][:3, :3].tolist())
                    )
                )
            )
            print("Trans. vec: ", A[0, idx][:3, 3], "\n")

        print("- Mean reprojection error")
        print(f" Overall: {np.mean(absolute_reproject_err):.5f}")  # Averaging reprojection errors over all images

        print("**** WARNING ****")
        print("MATLAB and python count array indices in a different way, i.e., python starts with 0 like a[0], a[1], "
              "... and so on, and Matlab goes like a(1), a(2), a(3), ..., and so on. Results shown here are based on"
              "pure MATLAB results. For further processing, small adjustment might be needed. For example, the center "
              "positions (cx, cy) have to be adjusted by 1 if you want to use the determined intrinsics matrix in "
              "python scripts.")

        plt.figure()
        plt.bar(
            np.arange(num_img_files), mean_abs_reproject_err, color="blue", alpha=0.5
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
