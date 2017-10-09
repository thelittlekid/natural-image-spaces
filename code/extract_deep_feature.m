function [ feature ] = extract_deep_feature( Iin, convnet, featureLayer )
% EXTRACT_DEEP_FEATURE: Extract deep feature for an image with pre-trained
% deep neural networks
%
% Author: Yifei Fan
% Email: yifei@gatech.edu
%
% Input: 
%   Iin: the input image;
%   convnet: the deep network you are using
%   featureLayer: specify the feature layer
% Output: 
%   feature: extracted deep feature with the specified deep network
% 
% Note: Require MATLAB version > 2016b
% Revision History: 
%   2017-06-14: Initial Version
%   2017-06-15: Input size for the first layer becomes adjustable 

inputSize = convnet.Layers(1).InputSize; % required input size for the 1st layer
I_resized = readAndPreprocessImage(Iin, inputSize);
feature = activations(convnet, I_resized, featureLayer, ...
                                'MiniBatchSize', 256, 'OutputAs', 'rows');

end

%% Utility Functions
% Resize the image to fit the requirement for AlexNet
% Note that other CNN models will have different input size constraints,
% and may require other pre-processing steps.
function Iout = readAndPreprocessImage(I, inputSize)

    % Some images may be grayscale. Replicate the image 3 times to
    % create an RGB image. 
    if ismatrix(I)
        I = cat(3,I,I,I);
    end

    % Resize the image as required for the CNN. 
    Iout = imresize(I, [inputSize(1) inputSize(2)]);  

    % Note that the aspect ratio is not preserved. In Caltech 101, the
    % object of interest is centered in the image and occupies a
    % majority of the image scene. Therefore, preserving the aspect
    % ratio is not critical. However, for other data sets, it may prove
    % beneficial to preserve the aspect ratio of the original image
    % when resizing.
end