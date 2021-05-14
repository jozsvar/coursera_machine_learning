% REGULARIZED LINEAR REGRESSION AND BIAS vs. VARIANCE - LINEAR REGRESSION COST FUNCTION 

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

% Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y. 
% Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% CALCULATE COST FUNCTION 
h = X * theta;

J_reg = (lambda/(2*m)) * sum(theta(2:end).^2);
jVal = (1/(2*m)) * (h - y)' * (h - y);
J = sum(jVal) + J_reg;

% CALCULATE GRADIENT DESCENT 
gradVal = X' * (h - y) / m;
grad_reg = (lambda/m) .* theta(2:end);
grad_reg = [0; grad_reg];
grad = gradVal + grad_reg;


grad = grad(:);

end
