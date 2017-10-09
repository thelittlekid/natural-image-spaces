%% Analyze CIFAR-10 classification results w.r.t. probability distribution
%
% Author: Yifei Fan (yifei@gatech.edu)

%% Set up
close all;
result_folder = '../result/';
addpath(result_folder);
load('testimages.mat');

% Load trained network and classification results for test samples, with
% specified number of random noise samples used during training.
noiserate = 0; % noise rate
input_file = [int2str(noiserate) 'noise.mat'];
load(input_file);

%% Main
% Calculate the probabilities for test samples and store them in a dictionary. 
% A GPU can accelerate the computation speed. 
probabilityMap = containers.Map();
Diff = (YTest ~= testLabels);
for i = 1:length(testLabels)
    feature = extract_deep_feature(testImages(:,:,:,i), cifar10Net, 'softmax');
    probabilityMap(char2str(1)) = feature;
    
    % TODO: Add calculation for robustness measurement
end