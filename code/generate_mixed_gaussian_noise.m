function [noiseImages] = generate_mixed_gaussian_noise(shape, mus, sigmas, ...
                                                        scorrs, filtermode)
% Generate a mixture of Gaussian noise sets under different settings
% This function calls sub-function: generate_gaussian_noise, with different
% controlling parameters on mean, variance and scale
% Input: 
%   shape: shape of the desired output, in vector form
%   mus: vector of means of the Gaussian noise groups, 1xn
%   sigma: vector of standard deviations of the Gaussian noise groups, 1xm
%   scorr: matrix of correlation scales used in the filter, 2xs
%   filtermode: 0 for medfilt2, 1 for imfilter
% Output:
%   noiseImages: generated mixed Gaussian noise images (h x w x d x noiseAmount)
%
% Author: Yifei Fan (yifei@gatech.edu)

noiseImages = uint8(zeros([shape(1:end-1), 0]));
noiseAmount = shape(end);
noiseAmountperGroup = round(noiseAmount/(length(mus)*length(sigmas)*size(scorrs,2)));

for mu = mus
    for sigma = sigmas
        for scorr = scorrs
            noiseGroup = generate_gaussian_noise([shape(1:end-1), noiseAmountperGroup], ...
                                     mu, sigma, scorr', filtermode);
            noiseImages = cat(4, noiseImages, noiseGroup);
        end
    end
end

end

