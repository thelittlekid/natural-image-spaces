function [indices] = get_classifiedimage_indices(imageset, labelset, net, rule)
% GET_CLASSIFIEDIMAGE_INDICES: Get the indices of images according to mode
% Input: 
%   imageset: a set of image
%   labelset: the label set of the image set
%   net: the neural-network classifier
%   rule: true-select correctly-classified images; false-misclassified
% Output:
%   indices: indices of the images that follow the rule 

labelTest = classify(net, imageset);
if rule
    indices = find(labelTest == labelset);
else
    indices = find(labelTest ~= labeset);
end

end

