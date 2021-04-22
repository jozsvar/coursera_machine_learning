% UNIVARIATE LINEAR REGRESSION - COST FUNCTION

function J = computeCost(X, y, theta)

%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;

prediction = X*theta;
sqrErrors = (prediction - y).^2;
J = 1/(2*m) * (sum(sqrErrors));


end