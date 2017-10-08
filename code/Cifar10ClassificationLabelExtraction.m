%% Examination of Test Results
% Take a care look on the misclassified samples, store them as well as
% their output labels and ground truth annotations. 
%
% Author: Yifei Fan (yifei@gatech.edu)
%% Set up
close all;
result_folder = '../result/';
addpath(result_folder);
load('testimages.mat'); % training and testing data with ground truth

% Load trained network and classification results for test samples, with
% specified number of random noise samples used during training. Then, set
% up the output folder. 
noiserate = 0; % noise rate
input_file = [int2str(noiserate) 'noise.mat'];
load(input_file);
% Create output folder
output_folder = [result_folder int2str(noiserate) 'noise/'];
mkdir(output_folder);

%% Store the misclassified samples and labels in output folder
Diff = (YTest ~= testLabels);
count = 0;

% text file recording labels
output_labels = 'labels.txt';
fout = fopen([output_folder output_labels], 'w'); % for recognition results

for i = 1:length(Diff)
    if(Diff(i))
        count = count + 1;
        img = testImages(:,:,:,i);
        remark = [int2str(i), ': ', ... 
                    'Ground truth-', char(testLabels(i)), ', ', ...
                    'Test result-', char(YTest(i)), '\n'];
        imwrite(img, [output_folder int2str(i) '.png']);
        fprintf(fout, remark);
    end
end

fclose('all');