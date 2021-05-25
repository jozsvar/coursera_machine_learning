%%%% ANOMALY DETECTION - GAUSSIAN PARAMETERS %%%%

function [mu sigma2] = estimateGaussian(X)
% The input X is the dataset with each n-dimensional data point in one row
% The output is an n-dimensional vector mu, the mean of the data set and the variances sigma^2, an n x 1 vector

% Useful variables (determine example size (m) and feature size (n)
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);


% loop through each coloumn and determine mu (=mean) and sigma2 (=variance)
for i = 1:n
    mu(i,:) = (1/m) * sum (X(:,i));
    sigma2(i,:) = (1/m) * sum((X(:,i) - mu(i,:)).^2);
end

end
