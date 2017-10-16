function [noiseImages] = generate_gaussian_noise(shape, mu, sigma, ... 
                                                    scorr, filtermode)
% Generate Gaussian random noise as an extra category in training
% Gaussian random noise is the most natural and general type of noise we
% could encounter. 
% We would like to generate Gaussian noise images under three controls:
% mean, variance and scale (used in mean value filter). 
% Note: the color of each pixel at each channel is i.i.d.. The only
% correlation is added via mean value filter, and the scale of such
% correlation is controlled by the scale argument
% Input: 
%   shape: shape of the desired output, in vector form
%   mu: mean of the Gaussian noise
%   sigma: standard deviation of the Gaussian noise
%   scorr: scale of correlation, used in the filter, 2 in a vector
%   filtermode: 0 for medfilt2, 1 for imfilter
% Output: 
%   noiseImages: generated Gaussian noise images (h x w x d x noiseAmount)
%
% Author: Yifei Fan (yifei@gatech.edu)

noiseImages = uint8(mu + sigma * randn(shape));
noiseAmount = shape(end);

if(isequal(scorr, [1; 1])) 
    return; % no need to filter if the pattern is 1x1
end

% Filter the noise images in specified mode
switch filtermode
    case 0
        % median filter
        for i = 1:noiseAmount
            r = medfilt2(noiseImages(:,:,1,i), scorr, 'symmetric'); 
            g = medfilt2(noiseImages(:,:,2,i), scorr, 'symmetric');
            b = medfilt2(noiseImages(:,:,3,i), scorr, 'symmetric');
            noiseImages(:,:,:,i) = cat(4, r, g, b);
        end
    case 1
        % mean value filter
        template = ones(scorr); % template for convolution
        template = template ./ sum(template(:)); % normalize (sum = 1)
        for i = 1:noiseAmount 
            noiseImages(:,:,:,i) = imfilter(noiseImages(:,:,:,i), template, 'circular');
        end
    otherwise
        % do not filter
end

end

