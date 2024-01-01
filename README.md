# About this Repository
This repository provides a Python script for getting camera intrinsics and extrinsics parameters using checkerboard images.
This script currently supports two different methods: **OpenCV-based** and **MATLAB-based** methods. 

# How to Use
The `main.py` code is very simple as shown below.  
<img src="./docs/images/input_example.png" width="1400">   
As the method, `CalibMethod.MATLAB` or `CalibMethod.OPENCV` can be selected. Once you specify the `CONFIG` dictionary, you can simply run this code. Please note that all input images should contain a checkerboard with good visibility and be stored in the `./data` folder.  

# What Can this Tool Do?
By using this tool, intrinsics and extrinsics parameters of your camera can be obtained. The extrinsics parameters can be obtained for each image.    
<img src="./docs/images/output_example.png" width="400">   
In addition, reprojection error can be also obtained in order to quantitatively assess the quality of the obtained camera calibration parameters.  
<img src="./docs/images/output_figure.png" width="500">  

# Available Methods
- **OpenCV-based method**: OpenCV functions are used to detect checkerboard corners and get camera intrinsics and extrinsics parameters. 
- **MATLAB-based method**: The MATLAB function, `./algorithm/matlab/calib_with_matlab.m`, is called from a python script and used to do the checkerboard detection and camera calibration.  

In general, these methods give almost the same results. If they are significantly different, there are several possible reasons: image quality affecting to detectability of checkerboard corners, difference of the position of the checkerboard origin (X=Y=Z=0 in space), or incorrect setting in the `CONFIG`.