% MULTIVARIATE LINEAR REGRESSION - FEATURE NORMALIZATION

function [X, mean, SD] = featureNormalize(X)

%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%set values (calculate mean an standard deviation (SD)) 
X_norm = X;
mean = mean(X);
SD = std(X);     

%elementwise division!
X_norm = (X - mean) ./ SD; 


end