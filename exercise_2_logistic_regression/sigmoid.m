% LOGISTIC REGRESSION - SIGMOID FUNCTION 

function g = sigmoid(z)

%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

g = (1 ./ (1 + exp(-z)));


end