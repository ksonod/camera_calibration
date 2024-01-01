function func(imageFileNames, squareSize)
    
    %% Detect checkerboard corners in images
    detector = vision.calibration.monocular.CheckerboardDetector();
    [imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
    imageFileNames = imageFileNames(imagesUsed);
    
    % Read the first image to obtain image size
    originalImage = imread(imageFileNames{1});
    [mrows, ncols, ~] = size(originalImage);
    
    % Generate world coordinates for the planar pattern keypoints
    worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);
    
    %% Calibrate the camera
    [cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
        'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);
    
    % Save intrinsics and distortion parameters
    save_file_path = [pwd, '/data/intrinsicsK.mat'];
    K = cameraParams.Intrinsics.K;
    save(save_file_path, 'K');

    save_file_path = [pwd, '/data/radialDistortion.mat'];
    rd = cameraParams.Intrinsics.RadialDistortion;
    save(save_file_path, 'rd');

    save_file_path = [pwd, '/data/tangentialDistortion.mat'];
    td = cameraParams.Intrinsics.TangentialDistortion;
    save(save_file_path, 'td');

    save_file_path = [pwd, '/data/extrinsicsA.mat'];
    A = {cameraParams.PatternExtrinsics.A};
    save(save_file_path, 'A');

    save_file_path = [pwd, '/data/reprojectionError.mat'];
    re = cameraParams.ReprojectionErrors;
    save(save_file_path, 're');

    save_file_path = [pwd, '/data/reprojectedPoints.mat'];
    rp = cameraParams.ReprojectedPoints;
    save(save_file_path, 'rp');

    save_file_path = [pwd, '/data/imagePoints.mat'];
    save(save_file_path, 'imagePoints');

    disp('Data saved.');
end