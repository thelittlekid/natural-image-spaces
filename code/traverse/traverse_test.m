%% Set up
% Load pretrained cifar10 network
load('rcnnStopSigns.mat','cifar10Net');
net = cifar10Net;
imageSize = net.Layers(1, 1).InputSize(1:end-1);
layer = net.Layers(end).Name;
m = net.Layers(end).OutputSize; % feature size: 10

% Load images: trainingImages(32x32x3x50000) and testImages(32x32x3x10000)
resultPath = '../../result/';
addpath(resultPath);
load('images.mat'); 

%% Configuration
tripnum = 10;
stopnum = 10;
imageset = testImages;
labelset = testLabels;

% A struct array that stores each traverse
%   StartImage, EndImage: index of the start and end image in the image set
%   StartLabel: label of the start and end image
%   PathLabels: labels of the images along the path that connects the two
%               images
T = struct('StartImage', {}, 'StartLabel', {}, 'EndImage', {}, 'EndLabel', {}, ...
           'PathLabels', {}, 'PathFeatures', {});
       
% Obtain indices of the images that are correctly classified
correctIndices = get_classifiedimage_indices(imageset, labelset, net, true);

%% Traverse
for i = 1:tripnum
    [Istart, Iend, Idxs, Idxe] = generate_image_pair(imageset, labelset, correctIndices, false);
    if Idxe == -1, continue; end % no pairs exist
    
    Iv = double(Iend) - double(Istart); % vector connecting Istart and Iend
    % pathLabels = categorical([]); % labels for each stop, categorical
    pathLabelstrs = strings(stopnum + 1, 1); % labels for each stop, string
    pathFeatures = zeros(stopnum + 1, m); % features for each stop
    
    for k = 0:stopnum
        Ik = uint8(double(Istart) + k/stopnum * Iv);
        Lk = classify(net, Ik);
        % pathLabels = cat(1, pathLabels, Lk);
        pathLabelstrs(k+1) = string(Lk);
        pathFeatures(k+1, :) = activations(net, Ik, layer);
    end
    
    % Store the path record into the struct array
    T(i).StartImage = Idxs;
    T(i).EndImage = Idxe;
    T(i).PathFeatures = pathFeatures;
    
    % Store the labels in categorical array
    % T(i).StartLabel = labelset(Idxs);
    % T(i).EndLabel = labelset(Idxe);
    % T(i).PathLabels = pathLabels;

    % Store labels to string for better display
    T(i).StartLabel = string(labelset(Idxs));
    T(i).EndLabel = string(labelset(Idxe));
    T(i).PathLabels = pathLabelstrs;
end

%% Save results
save([resultPath 'traverse_different.mat'], 'T');