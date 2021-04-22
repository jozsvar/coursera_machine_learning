% MULTIVARIATE LINEAR REGRESSION

%read comma seperated file
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);

%trainings examples 
m = length(y);

%print out values to check the data if normalization is needed
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%normalize features (save mean and SD for later)
[X, mean, SD] = featureNormalize(X);

% Add intercept term to X (=bias term)
X = [ones(m, 1) X];

% Run GRADIENT DESCENT
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

% Predict the price for 1650 sq-ft and 3 bedrooms (br)
% Normalize values with the calculated mu and sigma 
sq_ft=(1650-mu(:,1))./sigma(:,1);
br=(3-mu(:,2))./sigma(:,2);

price = [1, sq_ft, br] * theta;
fprintf('Predicted the price of a 1650 sq-ft, 3 bedroom house (using gradient descent):\n $%f', price);

clear;

% Run NORMAL EQUATION - no need to normalize the data 
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);

%trainings examples 
m = length(y);

% Add intercept term to X (=bias term)
X = [ones(m, 1) X];

theta = normalEqn(X, y);
fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));

% Predict the price for 1650 sq-ft and 3 bedrooms (br)

price = [1, 1650, 3]*theta;
fprintf('Predicted the price of a 1650 sq-ft, 3 bedroom house (using normal equation):\n $%f', price);





