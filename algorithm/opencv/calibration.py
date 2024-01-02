import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from visualization.checkerboard import show_cb_image_with_detected_corners, draw_XY_arrows
from algorithm.general.feature_analysis import define_XYZ_coordinate_system


def calibrate_with_opencv(config: dict, img_file_list: list):
    """
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    num_img_files = len(img_file_list)

    objp = np.zeros(
        (np.prod(config["checkerboard"]["num_corners"]), 3),
        np.float32
    )  # Object points in 3D
    objp[:, :2] = np.mgrid[
                  0:config["checkerboard"]["num_corners"][0],
                  0:config["checkerboard"]["num_corners"][1]
                  ].T.reshape(-1, 2) * config["checkerboard"]["checker_size"]  # Z values are always 0.

    points3d = []  # 3D point in real world space
    points2d = []  # 2D points in image plane.

    for idx, img_file in enumerate(img_file_list):
        img = np.array(Image.open(img_file))
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        detected, detected_corners = cv.findChessboardCorners(
            image=gray_img, patternSize=config["checkerboard"]["num_corners"], corners=None
        )

        if detected:
            points3d.append(objp)
            detected_corners_subpix = cv.cornerSubPix(
                image=gray_img, corners=detected_corners, winSize=(11, 11), zeroZone=(-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Termination criteria
            )
            points2d.append(detected_corners_subpix)
        else:
            print(f"Corners are not detected ({img_file.name})")
    cv.destroyAllWindows()

    ret, K, distortion_params, rvecs, tvecs = cv.calibrateCamera(
        objectPoints=points3d, imagePoints=points2d, imageSize=gray_img.shape, cameraMatrix=None, distCoeffs=None
    )

    np.set_printoptions(precision=3, suppress=True)
    print(f"- Intrinsic parameters : \n{K}")

    # See the link below to check the definition of distortion coefficients.
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    print(f" Distortion parameters (k1 k2 p1 p2 k3 k4 ...): {distortion_params}")

    print("- Extrinsic parameters")

    # Arrow setting for visualization
    magnification_factor = 30
    head_width = 15
    head_length = 10

    reprojection_error = []
    num_valid_files = len(points2d)  # Number of files where checkerboard corners are detected.
    for i in range(num_valid_files):

        if config["checkerboard"]["show_figure"]:
            plt.figure()

            img = np.array(Image.open(img_file_list[i]))

            # Show a checkerboard image with detected corners
            show_cb_image_with_detected_corners(
                img=img, detected_points=np.squeeze(points2d[i]), figure_title=img_file_list[i].name
            )

            # Set an origin (X, Y, Z) = (0, 0, 0) and unit vectors in X and Y directions.
            origin_point, x0, y0 = define_XYZ_coordinate_system(
                rvec=rvecs[i], tvec=tvecs[i], intrinsicK=K, distortion_coeff=distortion_params
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

        projected_points2d, _ = cv.projectPoints(
            objectPoints=points3d[i], rvec=rvecs[i], tvec=tvecs[i], cameraMatrix=K, distCoeffs=distortion_params
        )

        err = projected_points2d - points2d[i]

        # averaged over all the corners detected in a single image
        reprojection_error.append(
            np.mean(np.sqrt(err[:, :, 0] ** 2 + err[:, :, 1] ** 2))
        )

        print(f"{img_file_list[i].name} | Reprojection error = {reprojection_error[-1]:.5f}")
        Rt = np.concatenate([cv.Rodrigues(rvecs[i])[0], tvecs[i].reshape(3, 1)], axis=1)
        print(f"[R | t]:\n{Rt}")
        print("Rot. vec: ", rvecs[i].flatten(), "\n")

    print("- Mean reprojection error")
    print(f" Overall: {np.mean(reprojection_error):.5f}")  # Averaging reprojection errors over all images

    plt.figure()
    plt.bar(
        np.arange(num_img_files), reprojection_error, color="blue", alpha=0.5
    )
    plt.plot(
        [-0.5, num_img_files - 0.5], np.mean(reprojection_error) * np.ones(2), "k--"
    )
    plt.xlabel("Images")
    plt.ylabel("Mean reprojection error (pixel)")
    plt.show()
