# About this Repository
This repository provides a Python script for getting camera intrinsics and extrinsics parameters using checkerboard images.
This Python script currently supports two different methods: **OpenCV-based** [[1](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)] and **MATLAB-based** [[2](https://mathworks.com/help/vision/ref/estimatecameraparameters.html)] methods.

# How to Use
The `main.py` code is very simple as shown below:
<img src="./docs/images/input_example.png" width="1400">   
For now, `CalibMethod.MATLAB` or `CalibMethod.OPENCV` can be selected as a method. Once you specify the `CONFIG` dictionary, you can simply run this code. Please note that all input images should contain a checkerboard with good visibility and be stored in the `./data` folder.  

# What Can this Tool Do?
### 1. Camera Calibration
By using this tool, intrinsics and extrinsics parameters of a camera can be obtained. The extrinsics parameters can be determined for each image.    
<img src="./docs/images/output_example.png" width="400">    
Example data, which are images of a printed checkerboard pattern generated from [[3](https://calib.io/pages/camera-calibration-pattern-generator)], are provided in the `./data` folder. 

### 2. Clarifying the Coordinate System
In order to interpret the extrinsic parameters better, X and Y axes with a coordinate origin can be visualized together with detected checkerboard corners.  
<img src="./docs/images/output_example_checkerboard.png" width="200">    

### 3. Reprojection Error Evaluation
In addition, reprojection error can be also calculated in order to quantitatively assess the quality of the determined camera calibration parameters.  
<img src="./docs/images/output_figure.png" width="500">  

# Available Methods
- **OpenCV-based method**: OpenCV functions are used to detect checkerboard corners and get camera intrinsics and extrinsics parameters. 
- **MATLAB-based method**: The MATLAB function, `./algorithm/matlab/calib_with_matlab.m`, is called from a python script and used to do the checkerboard detection and camera calibration.  

In general, these methods give almost the same results. 
1. If extrinsics parameters are significantly different, it is most likely due to the difference of the position of the origin (i.e., X=Y=Z=0 in space). 
2. If intrinsics parameters are largely different, incorrect setting in the `CONFIG` or image quality affecting to detectability of checkerboard corners could be possible reasons.

# Technical Background
See also [[4](https://mathworks.com/help/vision/ug/camera-calibration.html)] and [[5](https://ieeexplore.ieee.org/document/888718)]. 

# References 
[[1](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)] OpenCV, *Camera Calibration*, opencv.org [Online]. Available: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html [Accessed: 01 January, 2024]  
[[2](https://mathworks.com/help/vision/ref/estimatecameraparameters.html)] MathWorks, *estimateCameraParameters*, mathworks.com [Online]. Available: https://mathworks.com/help/vision/ref/estimatecameraparameters.html [Accessed: 01 Januarry, 2024]  
[[3](https://calib.io/pages/camera-calibration-pattern-generator)] calib.io, *Pattern Generator*, calib.io [Online]. Available: https://calib.io/pages/camera-calibration-pattern-generator [Accessed: 02 January, 2024]  
[[4](https://mathworks.com/help/vision/ug/camera-calibration.html)] MathWorks, *What Is Camera Calibration*, mathworks.com [Online]. Available: https://mathworks.com/help/vision/ug/camera-calibration.html [Accessed: 02 January, 2024]  
[[5](https://ieeexplore.ieee.org/document/888718)] Z. Zhang, “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine Intelligence. vol. 22, no. 11, pp. 1330–1334, 2000.
