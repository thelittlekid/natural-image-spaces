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
noiserate = 3; % noise rate
input_file = [int2str(noiserate) 'noise.mat'];
load(input_file);

%% Main
% Calculate the probabilities for test samples and store them in a dictionary. 
% A GPU can accelerate the computation speed. 
probabilityMap = containers.Map();
Diff = (YTest ~= testLabels);
categoryName = cifar10Net.Layers(15).ClassNames;

count_testSamples = length(testLabels); % number of test samples (10,000)
count_diff = 0; % number of misclassified samples
count_same = 0; % number of correctly classified samples

% Robust measurements
confidence = 0; % confidence on correctly classified samples
dist_groundtruth_bound = 0; % distanlce to the nearest boundary
prob_groundtruth = 0; % probability for ground truth category when misclassified
mislead = 0; % confidence on the misclassification

for i = 1:length(testLabels)
    feature = extract_deep_feature(testImages(:,:,:,i), cifar10Net, 'softmax');
    probabilityMap(int2str(1)) = feature;
    
    % TODO: Add calculation for robustness measurement
    if(Diff(i))
        % for misclassified samples
        count_diff = count_diff + 1;
        % Get the ground truth label and calculate the distance to that
        % boundary
        idx = ismember(categoryName, char(testLabels(i)));
        prob_groundtruth = prob_groundtruth + feature(idx);
        mislead = mislead + max(feature);
    else
        % for correctly classified samples
        count_same = count_same + 1;
        confidence = confidence + max(feature);
    end
end

%%
if (count_same ~= 0)
    confidence = confidence/count_same;
end
if(count_diff ~= 0)
    prob_groundtruth = prob_groundtruth/count_diff;
    mislead = mislead/count_diff;
end