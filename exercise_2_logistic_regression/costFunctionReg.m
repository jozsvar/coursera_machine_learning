% REGULARIZED LOGISTIC REGRESSION - COST FUNCTION

function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Computes the cost of using theta as the parameter for regularized logistic regression and the gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);

J_reg = (lambda/(2*m)) * sum(theta(2:end).^2);
jVal = (-y' * log(h) - (1 - y')*log(1 - h)) /m;
J = sum(jVal) + J_reg;

% prepend a 0 column to our grad_reg matrix so we can use simple matrix addition
gradVal = X' * (h - y) / m;
grad_reg = (lambda/m) .* theta(2:end);
grad_reg = [0; grad_reg];
grad = gradVal + grad_reg;


end
