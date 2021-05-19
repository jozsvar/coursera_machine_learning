%%% PCA - DATA RECONSTRUCTION FROM COMPRESSED REPRESENTATION %%%

function X_rec = recoverData(Z, U, K)
% Recovers an approximation the original data that has been reduced to K dimensions. 
% It returns the approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));
               

% define U_reduce (k vectors onto which the data project the data) 
U_reduce = U(:, 1:K);

% calculate appoximated X
X_rec = Z * U_reduce';

end
