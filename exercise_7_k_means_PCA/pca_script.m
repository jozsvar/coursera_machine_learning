%%% PCA - SCRIPT %%%

% The following command loads the dataset. (variable X in environment)
load ('ex7data1.mat');

% Visualize the example dataset
figure;
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;

% IMPLEMENTING PCA % 
% Before running PCA, it is important to first normalize X (ensure every feature has zero mean)
[X_norm, mu, ~] = featureNormalize(X);

% Run PCA
[U, S] = pca(X_norm);

% Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector U(:,1) = %f %f \n', U(1,1), U(2,1));

% DIMENSIONALITY REDUCTION WITH PCA %
% Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));

% RECONSTRUCTION OF COMPRESSED DATA % 
% reconstruct the appoximate values from reduced data
X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));


% VISUALIZE THE PROJECTIONS %
%  Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square
%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

% FACE IMAGE DATASET %
% Load Face dataset (variable X loaded into the environment)
% This dataset was based on a cropped version of the labeled faces in the wild dataset.
load ('ex7faces.mat')

% Display the first 100 faces in the dataset
close all;
displayData(X(1:100, :));

% Run PCA on the face dataset:
% (1) normalize the dataset by subtracting the mean of each feature from the data matrix X. 
% (2) run PCA -> you will obtain the principal components of the dataset. Notice that each principal component in U (each row) is a vector of length n (where for the face dataset, n = 1024). 

% Normalize dataset
[X_norm, ~, ~] = featureNormalize(X);

% Run PCA
[U, ~] = pca(X_norm);

% Visualize these PCs by reshaping each of them into a 32x32 matrix that corresponds to the pixels in the original dataset
% Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

% Project the face dataset onto only the first 100 principal components. Concretely, each face image (z(i)) is now described by a 100 x 1 vector. 
K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: %d x %d', size(Z));

% To understand what is lost in the dimension reduction, you can recover the data using only the projected dataset
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;






