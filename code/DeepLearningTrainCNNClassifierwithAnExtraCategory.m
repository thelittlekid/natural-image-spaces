%% Training a deep network with an extra noise class
% This example is adapted from the "Object Detection Using Deep Learning"
% example from MATLAB. We delete the sections of stop sign detector and
% keep only the parts of training a classifier. We shut down all displays.
% The original comments are preserved for reference. 

%% Object Detection Using Deep Learning
% This example shows how to train an object detector using deep learning
% and R-CNN (Regions with Convolutional Neural Networks).
%
% Copyright 2016 The MathWorks, Inc.

%% Overview
% This example shows how to train an R-CNN object detector for detecting
% stop signs. R-CNN is an object detection framework, which uses a
% convolutional neural network (CNN) to classify image regions within an
% image [1]. Instead of classifying every region using a sliding window,
% the R-CNN detector only processes those regions that are likely to
% contain an object. This greatly reduces the computational cost incurred
% when running a CNN.
%
% To illustrate how to train an R-CNN stop sign detector, this example
% follows the transfer learning workflow that is commonly used in deep
% learning applications. In transfer learning, a network trained on a large
% collection of images, such as ImageNet [2], is used as the starting point
% to solve a new classification or detection task. The advantage of using
% this approach is that the pre-trained network has already learned a rich
% set of image features that are applicable to a wide range of images. This
% learning is transferable to the new task by fine-tuning the network. A
% network is fine-tuned by making small adjustments to the weights such
% that the feature representations learned for the original task are
% slightly adjusted to support the new task.
%
% The advantage of transfer learning is that the number of images required
% for training and the training time are reduced. To illustrate these
% advantages, this example trains a stop sign detector using the transfer
% learning workflow. First a CNN is pre-trained using the CIFAR-10 data
% set, which has 50,000 training images. Then this pre-trained CNN is
% fine-tuned for stop sign detection using just 41 training images.
% Without, pre-training the CNN, training the stop sign detector would
% require many more images.
%
% Note: This example requires Computer Vision System Toolbox(TM), Image
% Processing Toolbox(TM), Neural Network Toolbox(TM), and Statistics and
% Machine Learning Toolbox(TM).
%
% Using a CUDA-capable NVIDIA(TM) GPU with compute capability 3.0 or higher
% is highly recommended for running this example. Use of a GPU requires the
% Parallel Computing Toolbox(TM).

%% Download CIFAR-10 Image Data
% Download the CIFAR-10 data set [3]. This dataset contains 50,000 training
% images that will be used to train a CNN.

% Download CIFAR-10 data to a temporary directory
cifar10Data = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url, cifar10Data);

% Note: the above download may not work in MATLAB 2017b. If that is the
% case, you may manually download the cifar10 data from the above url, and
% place it in cifar10Data folder, e.g.cifar10Data = './';

% Load the CIFAR-10 training and test data.
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);

%% Add additional training samples
% Add random noise as a new category in the training data
% You can control the amount by changing the rate (1x = 5000 samples)
% noiseAmount = 5000 * 4; 
% noiseImages = uint8(randi([0 255], 32, 32, 3, noiseAmount));
% trainingImages = cat(4, trainingImages, noiseImages);
% noiseLabels = categorical(repmat({'noise'}, noiseAmount, 1));
% trainingLabels = cat(1, trainingLabels, noiseLabels);

% Add solid color as an additional category
% solidAmount = 5000*1;
% solidColors = uint8(randi([0 255], 3, solidAmount));
% solidImages = uint8(zeros(32, 32, 3, solidAmount));
% for i = 1:solidAmount
%     redChannel = repmat(solidColors(1, i), 32, 32);
%     greenChannel = repmat(solidColors(2, i), 32, 32);
%     blueChannel = repmat(solidColors(3, i), 32, 32);
%     solidImages(:,:,:,i) = cat(3, redChannel, greenChannel, blueChannel);
% end
% trainingImages = cat(4, trainingImages, solidImages);
% solidLabels = categorical(repmat({'solid'}, solidAmount, 1));
% trainingLabels = cat(1, trainingLabels, solidLabels);

% Add grayscale image as an additional category
% grayAmount = 5000 * 1;
% grayColors = uint8(randi([0 255], grayAmount));
% grayImages = uint8(zeros(32, 32, 3, grayAmount));
% for i = 1:grayAmount
%     grayImages(:,:,:,i) = grayColors(i);
% end
% trainingImages = cat(4, trainingImages, grayImages);
% grayLabels = categorical(repmat({'gray'}, grayAmount, 1));
% trainingLabels = cat(1, trainingLabels, grayLabels);

% % % TODO: Permutation

%%
% Each image is a 32x32 RGB image and there are 50,000 training samples.
size(trainingImages)

%%
% Select a subset of training samples to train the network
result_folder = '../result/';
addpath(result_folder);
load('retrain_indices.mat');

casenum = 3;
switch casenum
    case 1
        % low-low
        trainingImages = trainingImages(:,:,:,...
            [correct_lowconfidence_indices; incorrect_lowmislead_indices]); 
        trainingLabels = trainingLabels(...
            [correct_lowconfidence_indices; incorrect_lowmislead_indices]); 
    case 2
        % high-low
        trainingImages = trainingImages(:,:,:,...
            [correct_highconfidence_indices; incorrect_lowmislead_indices]);
        trainingLabels = trainingLabels(...
            [correct_highconfidence_indices; incorrect_lowmislead_indices]);
    case 3
        % low-high
        trainingImages = trainingImages(:,:,:,...
            [correct_lowconfidence_indices; incorrect_highmislead_indices]); 
        trainingLabels = trainingLabels(...
            [correct_lowconfidence_indices; incorrect_highmislead_indices]);
    case 4
        % high-high
        trainingImages = trainingImages(:,:,:,...
            [correct_highconfidence_indices; incorrect_highmislead_indices]); 
        trainingLabels = trainingLabels(...
            [correct_highconfidence_indices; incorrect_highmislead_indices]);
end

%%
% CIFAR-10 has 10 image categories. List the image categories:
% We add an additional category of random noise
numImageCategories = 10;
categories(trainingLabels)

%%

% Display a few of the training images, resizing them for display.
% figure
% thumbnails = trainingImages(:,:,:,1:100);
% thumbnails = imresize(thumbnails, [64 64]);
% montage(thumbnails)

%% Create A Convolutional Neural Network (CNN)
% A CNN is composed of a series of layers, where each layer defines a
% specific computation. The Neural Network Toolbox(TM) provides
% functionality to easily design a CNN layer-by-layer. In this example, the
% following layers are used to create a CNN:
%
% * |imageInputLayer|      - Image input layer
% * |convolutional2dLayer| - 2D convolution layer for Convolutional Neural Networks
% * |reluLayer|            - Rectified linear unit (ReLU) layer
% * |maxPooling2dLayer|    - Max pooling layer
% * |fullyConnectedLayer|  - Fully connected layer
% * |softmaxLayer|         - Softmax layer
% * |classificationLayer|  - Classification output layer for a neural network
%
% The network defined here is similar to the one described in [4] and
% starts with an |imageInputLayer|. The input layer defines the type and
% size of data the CNN can process. In this example, the CNN is used to
% process CIFAR-10 images, which are 32x32 RGB images:

% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)

%%
% Next, define the middle layers of the network. The middle layers are made
% up of repeated blocks of convolutional, ReLU (rectified linear units),
% and pooling layers. These 3 layers form the core building blocks of
% convolutional neural networks. The convolutional layers define sets of
% filter weights, which are updated during network training. The ReLU layer
% adds non-linearity to the network, which allow the network to approximate
% non-linear functions that map image pixels to the semantic content of the
% image. The pooling layers downsample data as it flows through the
% network. In a network with lots of layers, pooling layers should be used
% sparingly to avoid downsampling the data too early in the network.

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
% The first convolutional layer has a bank of 32 5x5x3 filters. A
% symmetric padding of 2 pixels is added to ensure that image borders
% are included in the processing. This is important to avoid
% information at the borders being washed away too early in the
% network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

% Note that the third dimension of the filter can be omitted because it
% is automatically deduced based on the connectivity of the network. In
% this case because this layer follows the image layer, the third
% dimension must be 3 to match the number of channels in the input
% image.

% Next add the ReLU layer:
reluLayer()

% Follow it with a max pooling layer that has a 3x3 spatial pooling area
% and a stride of 2 pixels. This down-samples the data dimensions from
% 32x32 to 15x15.
maxPooling2dLayer(3, 'Stride', 2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

]

%%
% A deeper network may be created by repeating these 3 basic layers.
% However, the number of pooling layers should be reduced to avoid
% downsampling the data prematurely. Downsampling early in the network
% discards image information that is useful for learning.
%
% The final layers of a CNN are typically composed of fully connected
% layers and a softmax loss layer.

finalLayers = [
    
% Add a fully connected layer with 64 output neurons. The output size of
% this layer will be an array with a length of 64.
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce 10 signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
classificationLayer
]

%%
% Combine the input, middle, and final layers.
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%%
% Initialize the first convolutional layer weights using normally
% distributed random numbers with standard deviation of 0.0001. This helps
% improve the convergence of training.

layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%% Train CNN Using CIFAR-10 Data
% Now that the network architecture is defined, it can be trained using the
% CIFAR-10 training data. First, set up the network training algorithm
% using the |trainingOptions| function. The network training algorithm uses
% Stochastic Gradient Descent with Momentum (SGDM) with an initial learning
% rate of 0.001. During training, the initial learning rate is reduced
% every 8 epochs (1 epoch is defined as one complete pass through the
% entire training data set). The training algorithm is run for 40 epochs.
%
% Note that the training algorithm uses a mini-batch size of 128 images. If
% using a GPU for training, this size may need to be lowered due to memory
% constraints on the GPU.

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

%% Train and test the accuracy for 50 times
accuracies = []; 
confidences = []; % expected confidence for correctly classified samples
misleads = []; % expected misleading probability for misclassified samples
prob_groundtruths = []; % probability for ground truth category when misclassified
noise_counts = []; % number of test samples that are misclassified to the extra category

for i = 1:50
    i
    %%
    % Train the network using the |trainNetwork| function. This is a
    % computationally intensive process that takes 20-30 minutes to complete.
    % To save time while running this example, a pre-trained network is loaded
    % from disk. If you wish to train the network yourself, set the
    % |doTraining| variable shown below to true.
    %
    % Note that a CUDA-capable NVIDIA(TM) GPU with compute capability 3.0 or
    % higher is highly recommeded for training.
    
    % A trained network is loaded from disk to save time when running the
    % example. Set this flag to true to train the network.
    doTraining = true;
    
    if doTraining
        % Train a network.
        cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
    else
        % Load pre-trained detector for the example.
        load('rcnnStopSigns.mat','cifar10Net')
    end
    
    %% Validate CIFAR-10 Network Training
    % After the network is trained, it should be validated to ensure that
    % training was successful. First, a quick visualization of the first
    % convolutional layer's filter weights can help identify any immediate
    % issues with training.
    
    % Extract the first convolutional layer weights
    w = cifar10Net.Layers(2).Weights;
    
    % rescale and resize the weights for better visualization
    w = mat2gray(w);
    w = imresize(w, [100 100]);
    
    % figure
    % montage(w)
    
    %%
    % The first layer weights should have some well defined structure. If the
    % weights still look random, then that is an indication that the network
    % may require additional training. In this case, as shown above, the first
    % layer filters have learned edge-like features from the CIFAR-10 training
    % data.
    %
    % To completely validate the training results, use the CIFAR-10 test data
    % to measure the classification accuracy of the network. A low accuracy
    % score indicates additional training or additional training data is
    % required. The goal of this example is not necessarily to achieve 100%
    % accuracy on the test set, but to sufficiently train a network for use in
    % training an object detector.
    
    % Run the network on the test set.
    YTest = classify(cifar10Net, testImages);
    
    % Calculate the accuracy.
    accuracy = sum(YTest == testLabels)/numel(testLabels)    
    accuracies = [accuracies, accuracy];
    
    % Calculate confidence and misleads, and prob_groundtruth
    [confidence, mislead, prob_groundtruth] = ...
        calculate_robustness_measurements(testImages, cifar10Net, YTest, testLabels);
    confidences = [confidences, confidence];
    misleads = [misleads, mislead];
    prob_groundtruths = [prob_groundtruths, prob_groundtruth];
    
    % Check if any test samples are misclassified as the extra category
    noise_count = sum(ismember(YTest, 'noise'));
    noise_counts = [noise_counts, noise_count];
end

disp("Average Test Accuray: " + mean(accuracies));
disp("Standard Deviation of Test Accuracy: " + std(accuracies));
disp("Average Confidence for Correctly Classified Samples: " + mean(confidence));
disp("Average Misleading Probability for Misclassified Samples: " + mean(misleads));
disp("Average Probability of Ground Truth Category for Misclassified Samples: " ...
    + mean(prob_groundtruths));
disp("Expected number of samples that are misclassified to the extra category: " ...
     + mean(noise_counts));
%%
% Further training will improve the accuracy, but that is not necessary for
% the purpose of training the R-CNN object detector.

%% Summary
% This example showed the effectiveness of the random samples in training a
% deep network.

%% References
% [1] Girshick, Ross, et al. "Rich feature hierarchies for accurate object
% detection and semantic segmentation." Proceedings of the IEEE conference
% on computer vision and pattern recognition. 2014.
%
% [2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image
% database." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
% Conference on. IEEE, 2009.
%
% [3] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of
% features from tiny images." (2009).
%
% [4] http://code.google.com/p/cuda-convnet/

% displayEndOfDemoMessage(mfilename)
