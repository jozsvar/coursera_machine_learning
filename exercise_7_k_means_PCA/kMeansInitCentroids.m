% K-MEANS - RANDOM INITIALIZATION OF CENTROIDS

function centroids = kMeansInitCentroids(X, K)
% returns K initial centroids to be used with the K-Means on the dataset X

% You should return this values correctly
centroids = zeros(K, size(X, 2));

%Randomly reorder the indicies of examples
randidx = randperm(size(X,1));

% Take the first K examples
centroids = X(randidx(1:K),:);

end