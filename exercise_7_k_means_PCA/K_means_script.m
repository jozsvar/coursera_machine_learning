% K-MEANS - SCRIPT

% Load an example dataset (X will be loaded to the environment)
load('ex7data2.mat');

% Select an initial set of centroids (K = 3 Centroids)
K = 3; 
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids);
fprintf('Closest centroids for the first 3 examples: %d %d %d', idx(1:3))

% Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);
fprintf('Centroids computed after initial finding of closest centroids: \n %f %f \n %f %f\n %f %f' , centroids);

% K-MEANS ON EXAMPLE DATASET 
% load dataset
load('ex7data2.mat');

% Settings for running K-Means
max_iters = 10;

% Select an initial set of centroids
initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
figure('visible','on'); hold on; 

% Plots the progress of the new means
plotProgresskMeans(X, initial_centroids, initial_centroids, idx, K, 1); 
xlabel('Press ENTER in command window to advance','FontWeight','bold','FontSize',14)

% calculates the new means
[~, ~] = runkMeans(X, initial_centroids, max_iters, true);
set(gcf,'visible','off'); hold off;

% K-MEANS ON PIXELS
%  Load an image of a bird
A = double(imread('bird_small.png'));
% Divide by 255 (because max value is 255) so that all values are in the range 0 - 1
A = A / 255; 

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels. 
% Each row will contain the Red, Green and Blue pixel values. 
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% initialize K and max_iters
K = 16;
max_iters = 10;

% determine initial centroids
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, ~] = runkMeans(X, initial_centroids, max_iters);

% finding the top K = 16 colors to represent the image, you can now assign each pixel position to its closest centroid
% by that represent the original image using the centroid assignments of each pixel

% represent the image X as in terms of the indices in idx.
% Recover the image from the indices (idx) by mapping each pixel (specified by it's index in idx) to the centroid value.
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
figure;
subplot(1, 2, 1);
imagesc(A); 
title('Original');
axis square

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
axis square


