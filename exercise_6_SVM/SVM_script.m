% SUPPORT VECTOR MACHINE - SCRIPT

%Load data set 1 of exercise 6 (X, y will be in the environment)
load('ex6data1.mat');

% Plot training data
plotData(X, y);


% use SVM software packages (most SVM software packages (including svmTrain.m) automatically add the extra feature x0 = 1)
% with C = 1, SVM puts the decision boundary in the gap between the two datasets
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

% with C = 100, SVM classifies every single example correctly, but has a decision boundary that does not appear to be a natural fit for the data
C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);


% use SVM with gaussian kernels to do non-linear classification
x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);



%Load data set 2 of exercise 6 (X, y will be in the environment)
load('ex6data2.mat');

% Plot training data
plotData(X, y);

% SVM parameters 
C = 1;
sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. 
% However, in practice, you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
% The decision boundary is able to separate most of the positive and negative examples correctly and follows the contours of the dataset well
visualizeBoundary(X, y, model);


%Load data set 3 of exercise 6 (X, Xval, y, yval will be in the environment)
load('ex6data3.mat');

% Plot training data
plotData(X, y);

% determine the best C and sigma parameters to use
% Try different SVM Parameters here values = 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
