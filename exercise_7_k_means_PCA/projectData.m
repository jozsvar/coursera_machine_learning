%%% PCA - PROJECT DATA INTO REDUCED DIMENSIONAL SPACE %%%

function Z = projectData(X, U, K)
% Computes the projection of the normalized inputs X into the reduced dimensional space spanned by the first K columns of U. 
% It returns the projected examples in Z.


% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% define U_reduce (k vectors onto which the data project the data) 
U_reduce = U(:, 1:K);

% project the data 
Z = X * U_reduce;

end