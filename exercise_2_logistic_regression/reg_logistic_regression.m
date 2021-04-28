% REGULARIZED LOGISTIC REGRESSION

% Load data
data = load('ex2data2.txt');
% Load variables
X = data(:, [1, 2]); 
y = data(:, 3);

% PLOT DATA
plotData(X, y);
hold on;

% Labels and Legend
xl = xlabel('Microchip Test 1');
% move x-label to position -0.95 (corresponds to y axis) - 2 = vertical
xl.Position(2) = -0.95; 

yl = ylabel('Microchip Test 2');
% move x-label to position -1.2 (corresponds to x axis) - 1 = horizontal
yl.Position(1) = -1.2;

% Specified in plot order
legend({'y = 1', 'y = 0'}, 'Location', 'best')
hold off;

% FEATURE MAPPING
% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));

% COST FUNCTION AND GRADIENT
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

% Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);
fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

%LEARNING PARAMETERS USING fminunc
%  Set options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% PLOT DATA WITH DECISION BOUNDRY
plotDecisionBoundary(theta, X, y);
hold on;

% Labels and Legend
xl = xlabel('Microchip Test 1');
% move x-label to position -1.2 (corresponds to y axis) - 2 = vertical
xl.Position(2) = -1.2; 

yl = ylabel('Microchip Test 2');
% move x-label to position -1.2 (corresponds to x axis) - 1 = horizontal
yl.Position(1) = -1.2;

% Specified in plot order
legend({'y = 1', 'y = 0', 'Decision Boundary'}, 'Location', 'best')
hold off;

