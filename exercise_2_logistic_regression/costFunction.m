% LOGISTIC REGRESSION - COST FUNCTION 

function [J, grad] = costFunction(theta, X, y)

% Computes the cost of using theta as the parameter for logistic regression and the gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);

J = (-y' * log(h) - (1 - y')*log(1 - h)) / m;
grad = X' * (h - y) / m;


end