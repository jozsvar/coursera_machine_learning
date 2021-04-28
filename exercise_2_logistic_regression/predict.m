%LOGISTIC REGRESSION - PREDICTION FUNCTION

function p = predict(theta, X)

% Computes the predictions for X using a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% Number of training examples
m = size(X, 1); 

% You need to return the following variables correctly
p = zeros(m, 1);

p = sigmoid(X * theta) >= 0.5;


end