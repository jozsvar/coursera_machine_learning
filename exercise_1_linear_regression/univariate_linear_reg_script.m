% UNIVARIATE LINEAR REGRESSION 
 

% read comma separated data
data = load('ex1data1.txt'); 
X = data(:, 1); %all values from coloumn 1
y = data(:, 2); %all values from coloumn 2

%scatter plot with values from data
plotData(X,y)

%define values
m = length(X); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters = [0;0]
iterations = 1500;
alpha = 0.01;

% Compute and display initial cost with theta all zeros
computeCost(X, y, theta)

% Run gradient descent:
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

% Plot the linear fit
hold on; % keep previous scatter plot with values from data
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35 000 and 70 000
prediction1 = [1, 3.5] *theta;
fprintf('For population size 35 000, we predict a profit of %f\n', prediction1*10000);
prediction2 = [1, 7] * theta;
fprintf('For population size 70 000, we predict a profit of %f\n', prediction2*10000);


