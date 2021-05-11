% NEURAL NETWORK SCRIPT

% Load file (data has been saved in a native MATLAB matrix format)
% The matrices X and y will already be loaded to the MATLAB environment
load('ex3data1.mat');

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
% Select 100 data points
sel = sel(1:100);
%display in a matrix
displayData(X(sel, :));

% Load set of network parameters that were already trained before
% Loaded into the MATLAB environment: Theta1 has size 25 x 401, Theta2 has size 10 x 26
% 25 units in the second layer and 10 output units
load('ex3weights.mat'); 

% FEEDWORWARD PROPAGATION AND PREDICTION
% Implement the feedforward computation that computes 'h_theta(x^(i))' for every example 'i' and returns the associated predictions

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% DISPLAY IMAGES from the training set one at a time, while the console prints out the predicted label for the displayed image. 
%  Randomly swap examples
rp = randi(m);

pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

% Display image
displayData(X(rp, :));




