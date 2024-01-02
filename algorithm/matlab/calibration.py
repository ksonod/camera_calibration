import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from PIL import Image
from visualization.checkerboard import show_cb_image_with_detected_corners, draw_XY_arrows
from algorithm.general.feature_analysis import define_XYZ_coordinate_system

def calibrate_with_matlab(config: dict, img_file_list: list):
    """
    Tested with MATLAB R2023b + python 3.10
    """
    import matlab.engine
    print("Start a MATLAB engine...")
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

        A, K, radial_distortion, tangential_distortion, reprojection_error, points2d = load_mat_files(data_folder)
        absolute_reproject_err = np.sqrt(reprojection_error[:, 0, :] ** 2 + reprojection_error[:, 1, :] ** 2)

        # Reprojection error is averaged over all the detected points for each image
        mean_abs_reproject_err = np.mean(absolute_reproject_err, axis=0)

        print("Finished processing data... \n")

        np.set_printoptions(precision=3, suppress=True)
        print(f"- Intrinsic parameters : \n{K}")
        print(f"  Radial distortion k: {radial_distortion}")
        print(f"  Tangential distortion p: {tangential_distortion}\n")
        print("- Extrinsic parameters")

        # Arrow setting for visualization
        magnification_factor = 30
        head_width = 15
        head_length = 10

        for idx_file in range(num_img_files):

            if config["checkerboard"]["show_figure"]:
                plt.figure()

                img = np.array(Image.open(img_file_list[idx_file]))

                show_cb_image_with_detected_corners(
                    img=img, detected_points=points2d[:, :, idx_file], figure_title=img_file_list[idx_file].name
                )

                rvec = np.array(
                    eng.rotmat2vec3d(
                        matlab.double(A[0, idx_file][:3, :3].tolist())
                    )
                )
                tvec = A[0, idx_file][:3, 3]

                # Prepare distortion coefficients in the format of OpenCV (k1 k2 p1 p2 k3 k4 ...)
                if radial_distortion[0].shape[0] == 2:  # MATLAB's default
                    distortion_coeff = np.concatenate([radial_distortion[0], tangential_distortion[0]])
                elif radial_distortion[0].shape[0] == 3:
                    distortion_coeff = np.concatenate(
                        [
                            radial_distortion[0][:2],
                            tangential_distortion[0],
                            np.array(
                                [radial_distortion[0][2]]
                            )
                        ]
                    )

                # Set an origin (X, Y, Z) = (0, 0, 0) and unit vectors in X and Y directions.
                origin_point, x0, y0 = define_XYZ_coordinate_system(
                    rvec=rvec, tvec=tvec, intrinsicK=K, distortion_coeff=distortion_coeff
                )

                # Draw arrows to show X and Y axes
                draw_XY_arrows(
                    origin_point=origin_point,
                    x0=x0,
                    y0=y0,
                    magnification_factor=magnification_factor,
                    head_width=head_width,
                    head_length=head_length,
                )

            print(f"{img_file_list[idx_file].name} | Reprojection error = {mean_abs_reproject_err[idx_file]:.5f}")

            print(f"[R | t]: \n{A[0, idx_file][:3, :]}")
            print(
                "Rot. vec: ",
                np.array(
                    eng.rotmat2vec3d(
                        matlab.double(A[0, idx_file][:3, :3].tolist())
                    )
                ),
                "\n"
            )

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

    # Get detected corners in 2D
    dp = loadmat(data_folder / "imagePoints.mat")
    points2d = dp["imagePoints"]  # [number of detected points in a single image] x 2 x [number of images]

    return A, K, radial_distortion, tangential_distortion, reprojection_error, points2d
