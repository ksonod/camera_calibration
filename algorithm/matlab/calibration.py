import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from PIL import Image
from visualization.checkerboard import show_cb_image_with_detected_corners, draw_XY_arrows
from algorithm.general.feature_analysis import define_XYZ_coordinate_system
from algorithm.general.calib import CameraCalib


def calibrate_with_matlab(config: dict, img_file_list: list):
    """
    Tested with MATLAB R2023b + python 3.10
    """

    matlabcalib = MatlabCalib(config, img_file_list)
    matlabcalib()


class MatlabCalib(CameraCalib):
    """
    Checkerboard corner detection and camera calibration (extrinsics and intrinsics) are done with MATLAB. The MATLAB
    file calib_with_matlab.m includes a function to handle these data processing and generates some *.mat files in the
    data folder so that those files can be read in this python script.

    This was tested with MATLAB R2023b + python 3.10
    """

    def __init__(self, config: dict, img_file_list: list):
        super().__init__(config, img_file_list)

    def __call__(self):
        import matlab.engine
        print("Start a MATLAB engine in python...")
        eng = matlab.engine.start_matlab()
        eng.addpath("algorithm/matlab")

        str_img_file_list = eng.cell(1, self.num_img_data)

        # Convert the file list type for MATLAB
        for idx, path_file in enumerate(self.img_file_list):
            str_img_file_list[idx] = self.img_file_list[idx].absolute().__str__()

        # Run matlab script for calibration and save calibration data in .mat files in ./data folder.
        print("Run a MATLAB calibration script...")
        try:
            eng.calib_with_matlab(str_img_file_list, self.checker_size, nargout=0)
        except:
            print("MATLAB did not run successfully.")

        data_folder = Path.cwd() / "data"  # Path to a directory containing generated *.mat files

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

            for idx_file in range(self.num_img_data):

                if self.show_figure:
                    plt.figure()

                    img = np.array(Image.open(self.img_file_list[idx_file]))

                    show_cb_image_with_detected_corners(
                        img=img, detected_points=points2d[:, :, idx_file],
                        figure_title=self.img_file_list[idx_file].name,
                        marker_style="x", marker_color="yellow", label="Detected corners"
                    )
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.tight_layout()

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

                print(f"{self.img_file_list[idx_file].name} | Reprojection error = {mean_abs_reproject_err[idx_file]:.5f}")

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
            print("MATLAB and python count array indices in a different way. Obtained parameters are not adjusted.")

            plt.figure()
            plt.bar(
                np.arange(self.num_img_data), mean_abs_reproject_err, color="blue", alpha=0.5
            )
            plt.plot(
                [-0.5, self.num_img_data - 0.5], np.mean(absolute_reproject_err) * np.ones(2), "k--"
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
