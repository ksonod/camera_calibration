import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def calibrate_with_opencv(config: dict, img_file_list: list):
    """
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    num_img_files = len(img_file_list)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(config["checkerboard"]["num_corners"]), 3), np.float32)
    objp[:, :2] = np.mgrid[
                  0:config["checkerboard"]["num_corners"][0],
                  0:config["checkerboard"]["num_corners"][1]
                  ].T.reshape(-1, 2) * config["checkerboard"]["checker_size"]
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    gray_img_stack = []
    for idx, img_file in enumerate(img_file_list):
        img = np.array(Image.open(img_file))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, config["checkerboard"]["num_corners"], None)
        # If found, add object points, image points (after refining them)
        if ret:
            gray_img_stack.append(gray)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if config["checkerboard"]["show_figure"]:
                cv.drawChessboardCorners(img, config["checkerboard"]["num_corners"], corners2, ret)
                cv.imshow('img', img)
                cv.waitKey()
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.set_printoptions(precision=3, suppress=True)
    print(f"- Intrinsic parameters : \n{mtx}\n")

    print("- Extrinsic parameters")

    reprojection_error = []
    for i in range(num_img_files):
        projected_points2d, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # averaged over all the corners detected in a single image
        reprojection_error.append(
            cv.norm(imgpoints[i], projected_points2d, cv.NORM_L2) / len(projected_points2d)
        )

        print(f"{img_file_list[i].name} | Reprojection error = {reprojection_error[-1]:.5f}")
        print("Rot. vec: ", rvecs[i].flatten())
        print("Trans. vec: ", tvecs[i].flatten(), "\n")
        # Rt = np.concatenate([cv.Rodrigues(rvecs[0])[0], tvecs[0].reshape(3,1)], axis=1)
        # print(f"[R | t]:\n{Rt}\n")

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

