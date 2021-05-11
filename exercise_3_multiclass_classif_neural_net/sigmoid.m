% MULTI-CLASS CLASSIFICATIONS AND NEURAL NETWORKS - SIGMOID FUNCTION

function g = sigmoid(z)
% Compute sigmoid functoon


g = 1 ./ (1 + exp(-z));


end