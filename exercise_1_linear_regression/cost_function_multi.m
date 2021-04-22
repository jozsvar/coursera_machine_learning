% MULTIVARIATE LINEAR REGRESSION - COST FUNCTION

function J = computeCostMulti(X, y, theta)

%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;

J=(1/(2*m)) * (X * theta - y)' * (X * theta - y);

end