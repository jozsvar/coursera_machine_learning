% NEURAL NETWORK SCRIPT - FEEDWORWARD PROPAGATION AND PREDICTION

function p = predict(Theta1, Theta2, X)
%   The function outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%FEEDWORWARD PROPAGATION
%hidden layer in layer 2
z2 = X * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];

%output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%PREDICTION
[p_max, p] = max(a3, [], 2);   


end