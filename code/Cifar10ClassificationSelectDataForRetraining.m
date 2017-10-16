%% Select subgroups of training data to retrain the network
% Retrain the network with samples that are far from the decision
% boundaries and those that are close to the decision boundaries, to see
% the impact of training samples on test accuracy.
% We assume that the probability in softmax layer is correlated to the
% distance to decision boundaries. 
% This script does the first step, that is, to select groups of training
% samples. You have to rerun the training process using: 
%   DeepLearningTrainCNNClassifierwithAnExtraCategory.m
% This might seem to be a little unorthodox, as the distance can be
% obtained only after one complete training process, it can't help the
% training process after all. The key idea here is to illustrate that the
% "value" of training samples might be different. 
%
% Author: Yifei Fan (yifei@gatech.edu)

%% Set up
close all;
result_folder = '../result/';
addpath(result_folder);

load('images.mat');

% Load pretrained network and classification results for test samples, with
% specified number of random noise samples used during training. 
% The pretrained network is used to select samples for the retraining process. 
noiserate = 0;
input_file = [int2str(noiserate) 'noise.mat'];
load(input_file);


%% Calculate confidence and misleading probability
confidences = zeros(length(trainingLabels), 1); % expected confidence for correctly classified samples
misleads = zeros(length(trainingLabels), 1); % expected misleading probability for misclassified samples
YTrain = classify(cifar10Net, trainingImages); % labels for training samples with pretrained network

Diff = (YTrain ~= trainingLabels);
incorrect_count = sum(Diff);
correct_count = length(Diff) - incorrect_count;

for i = 1:length(trainingLabels)
    feature = extract_deep_feature(trainingImages(:,:,:,i), cifar10Net, 'softmax');
    prob = max(feature);
    % Check if the calculated labels for training samples match the ground truth
    % If yes, confidence denotes the real confidence
    % If not, probability for ground truth category reflects the distance to the ground truth category.
    if(Diff(i))
        % for misclassified samples
        misleads(i) = prob;
    else
        % for correctly classified samples
        confidences(i) = prob;
    end
end

%% Sort and select a subset of trianing samples for retraining 
% Obtain the training samples that are categorized with high/low probabilities
percentage = 0.5; % percentage of training samples that will be used for retraining
if percentage < 0 || percentage > 1
    error('percentage must between 0 and 1');
end

correct_num = round(correct_count*percentage);
incorrect_num = round(incorrect_count*percentage);

% Sort the confidences and misleads
[~, correct_confidence_indices] = sort(confidences);
[~, incorrect_mislead_indices] = sort(misleads);

% Obtain 4 groups of indices for samples with high/low confidence
correct_lowconfidence_indices = correct_confidence_indices(1:correct_num);
correct_highconfidence_indices = correct_confidence_indices(end-correct_num+1:end);
incorrect_lowmislead_indices = incorrect_mislead_indices(1:incorrect_num);
incorrect_highmislead_indices = incorrect_mislead_indices(end-incorrect_num+1:end);

save([result_folder, 'retrain_indices.mat'], ...
    'correct_highconfidence_indices', 'correct_lowconfidence_indices', ...
    'incorrect_lowmislead_indices', 'incorrect_highmislead_indices');