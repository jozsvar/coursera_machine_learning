% MULTI-CLASS CLASSIFICATIONS

% Load file (data has been saved in a native MATLAB matrix format)
% The matrices X and y will already be loaded to the MATLAB environment
load('ex3data1.mat');

% VISUALIZING A SUBSET OF THE TRAINING SET
% Code randomly selects 100 rows from X and passes those rows to the displayData function 
% This function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

% VECTORIZING LOGISTIC REGRESSION
% TEST the vectorized implementation and compare to expected outputs
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;

[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

% print COST
fprintf('Cost: %f | Expected cost: 2.534819\n',J);

% print GRADIENT
fprintf('Gradients:\n'); 
fprintf('%f\n',grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');

% ONE-VS-ALL CLASSIFICATIONS
% Train a multi-class classifier
% 10 labels, from 1 to 10
num_labels = 10;  
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% Predict the digit contained in a given image
% The function will pick the class for which the corresponding logistic regression classifier outputs the highest probability

pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);




