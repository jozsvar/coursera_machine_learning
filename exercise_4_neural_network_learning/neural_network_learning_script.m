% NEURAL NETWORK LEARNING SCRIPT

% Load data (loads X and y into the working evironment)
load('ex4data1.mat');
% number of training examples
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel, :));

% MODEL REPRESENTATION
% Since the images are of size 20 x 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1)

% Load a set of network parameters already trained - loads Theta1 and Theta2
% The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes)
load('ex4weights.mat');

% FEEDFORWARD AND COST FUNCTION 
% 20x20 Input Images of Digits
input_layer_size  = 400; 
% 25 hidden units 
hidden_layer_size = 25;
% 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10;          

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

fprintf('Cost at parameters (loaded from ex4weights): %f', J);

% REGULARIZED COST FUNCTION 
% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters (loaded from ex4weights): %f', J);

%BACKPROPAGATION
% Call your sigmoidGradient function
sigmoidGradient(0)

% Random initialization
% When training neural networks, it is important to randomly initialize the parameters for symmetry breaking. 
% One effective strategy for random initialization is to randomly select values for Theta(l) uniformly in the range [-epsilon, epsilon]. 
% You should use epsilon = 0.12. This range of values ensures that the parameters are kept small and makes the learning more efficient.

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% BACKPROPAGATION
% The function checkNNGradients.m will create a small neural network and dataset that will be used for checking your gradients. 
% If your backpropagation implementation is correct, you should see a relative dierence that is less than 1e-9. (value = 2.2366e-11)
checkNNGradients;

% REGULARIZED NEURAL NETWORK
%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging value 
% This value should be about 0.576051
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at (fixed) debugging parameters (w/ lambda = 3): %f', debug_J);

% LEARNING PARAMETERS USING fmincg

options = optimset('MaxIter', 50);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Predict the training accuracy (value = 95.94%)
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% VISUALIZING THE HIDDEN LAYER
% Visualize Weights 
displayData(Theta1(:, 2:end));


