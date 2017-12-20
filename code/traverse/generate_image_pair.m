function [Is, Ie, Idxs, Idxe] = generate_image_pair(imageset, labelset, correctIndices, mode)
% GENERATE_IMAGE_PAIR: Generate a image pair for traverse
% Note: the imageset should contain images that are correctly classified
% Input: 
%   imageset: image set, hxwxdxn uint8
%   labelset: category set, nx1 categorical
%   correctIndices: indices of images that are correctly classified
%   mode: false-images from different categories, true-from the same category
% Output: 
%   Is, Ie: start and end image
%   Idxs, Idxe: index of the start and end image

n = length(correctIndices); % number of images that are correctly classified

% Randomly pick a start image
Idxs = correctIndices(randi(n)); 
Is = imageset(:,:,:,Idxs);
Ls = labelset(Idxs);


if mode
    % Images from the same cateogory
    imagepool = intersect(find(labelset == Ls), correctIndices);
    imagepool = imagepool(imagepool ~= Idx); % remove the start index
else
    % Images from different categories
    imagepool = intersect(find(labelset ~= Ls), correctIndices);
end

if isempty(imagepool) % No feasible end images exist
    Idxe = -1;
    return;
end

% Randomly select the end image from the image pool
Idxe = imagepool(randi(length(imagepool)));
Ie = imageset(:,:,:,Idxe);

end

