% MULTI-CLASS CLASSIFICATIONS - REGULARIZED LOGISTIC REGRESSION - COST FUNCTION - GRADIENT DESCENT

function [J, grad] = lrCostFunction(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% CALCULATE COST FUNCTION 
h = sigmoid(X * theta);

J_reg = (lambda/(2*m)) * sum(theta(2:end).^2);
jVal = (-y' * log(h) - (1 - y')*log(1 - h)) /m;
J = sum(jVal) + J_reg;

% CALCULATE GRADIENT DESCENT 
gradVal = X' * (h - y) / m;
grad_reg = (lambda/m) .* theta(2:end);
grad_reg = [0; grad_reg];
grad = gradVal + grad_reg;


end
