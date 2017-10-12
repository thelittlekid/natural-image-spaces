function [confidence, mislead, prob_groundtruth] = calculate_robustness_measurements(samples, net, results, groundtruth)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%
% Output: 
%   confidence: expected confidence of correctly classified samples
%   mislead: expected max probability for misclassified samples
%   prob_groundtruth: probability for ground truth category when misclassified

confidence = 0; 
mislead = 0;
prob_groundtruth = 0;
count_diff = 0;
count_same = 0;

sample_num = min([size(samples, 4), length(results), length(groundtruth)]);
categoryName = net.Layers(15).ClassNames;
Diff = (results ~= groundtruth);
for i = 1:sample_num
    feature = extract_deep_feature(samples(:,:,:,i), net, 'softmax');
    if(Diff(i))
        % misclassified samples
        count_diff = count_diff + 1;
        % Get the ground truth label and calculate the distance to that
        % boundary
        idx = ismember(categoryName, char(groundtruth(i)));
        prob_groundtruth = prob_groundtruth + feature(idx);
        mislead = mislead + max(feature);
    else
        % for correctly classified samples
        count_same = count_same + 1;
        confidence = confidence + max(feature);
    end
end

if (count_same ~= 0)
    confidence = confidence/count_same;
end
if(count_diff ~= 0)
    prob_groundtruth = prob_groundtruth/count_diff;
    mislead = mislead/count_diff;
end

end

