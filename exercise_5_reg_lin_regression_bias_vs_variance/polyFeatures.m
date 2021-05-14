% REGULARIZED LINEAR REGRESSION AND BIAS VS VARIANCE - POLYNOMIAL FEAUTRES

function [X_poly] = polyFeatures(X, p)

% Takes a data matrix X (size m x 1) and maps each example into its polynomial features where
% X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

for i = 1:p
    X_poly(:, i) = X.^i;
end

end
