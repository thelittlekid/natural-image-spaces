%% Find unique misclassified samples of two tests
% Given two arrays of test results, we extract the unique misclassified
% samples for each test. E.g. samples that were misclassified in test A
% but correctly classified in test B, will be counted as a unique
% misclassified sample for test A. We do this to reveal the different
% efforts spent by two given classifiers.  
%
% Author: Yifei Fan (yifei@gatech.edu)

%% Set up
close all;
result_folder = '../result/';
addpath(result_folder);
load('0vs3.mat');

rateA = 0; rateB = 3; % noise rates
rates = [rateA, rateB]; rateA = min(rates);  rateB = max(rates); % make sure rateA < rateB

% Create output folder
output_folder = [result_folder int2str(rateA) 'vs' int2str(rateB) '/'];
mkdir(output_folder);

%%  
output_A = [int2str(rateA) '.txt']; % unique misclassified samples for classifer A
output_B = [int2str(rateB) '.txt']; % unique misclassified samples for classifier B
output_common = ['common' '.txt'];  % common samples that are misclassified by both

foutA = fopen([output_folder output_A], 'w');
foutB = fopen([output_folder output_B], 'w');
foutcommon = fopen([output_folder output_common], 'w');

% The names of output arrays follow the pattern YTest[noiserate]. To obtain
% the name of corresponding array, we use 'eval' to execute the string as a
% variable name
labelA = eval(['YTest' int2str(rateA)]);
labelB = eval(['YTest' int2str(rateB)]);
DiffA = (labelA ~= testLabels);
DiffB = (labelB ~= testLabels);
for i = 1:length(testLabels)
    if(DiffA(i))
        if(DiffB(i)) % common
            remark = [int2str(i), ': ', ...
                        'Ground truth-', char(testLabels(i)), ', ', ...
                        'Test result A-', char(labelA(i)), ', ', ...
                        'Test result B-', char(labelB(i)), '\n']; 
            fprintf(foutcommon, remark); 
        else  % unique A
            remark = [int2str(i), ': ', ...
                        'Ground truth-', char(testLabels(i)), ', ', ...
                        'Test result A-', char(labelA(i)), '\n'];
            fprintf(foutA, remark);
        end
    else
        if(DiffB(i)) % unique B
            remark = [int2str(i), ': ', ...
                        'Ground truth-', char(testLabels(i)), ', ', ...
                        'Test result B-', char(labelB(i)), '\n'];
            fprintf(foutB, remark);
        end
    end
end

fclose('all');