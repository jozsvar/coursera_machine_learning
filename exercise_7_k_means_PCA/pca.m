%%% PCA - PCA ALGORITHM %%%

function [U, S] = pca(X)
% computes eigenvectors of the covariance matrix of X 
% Returns the eigenvectors U, the eigenvalues (on diagonal) in S

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% compute covariance matrix Sigma 
Sigma = (1/m) * X' * X;

% compute eigenvector of matrix Sigma with singular value decomposition
[U, S, V] = svd(Sigma);


end
