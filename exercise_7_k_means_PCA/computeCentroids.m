% K-MEANS - CALCULATE NEW CENTROID

function centroids = computeCentroids(X, idx, K)
% returns the new centroids by computing the means of the data points assigned to each centroid. 
% It is given a dataset X where each row is a single data point, a vector idx of centroid assignments 
% (i.e. each entry in range [1..K]) for each example, and K, the number of centroids. 

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% loops through K
for k = 1:K

    % initialize parameters
    number_k = 0;
    sum = zeros(n, 1);

    % loop through examples
    for i = 1:m
        % if the ith position in idk is the same as k, sum the numbers
        if idx(i) == k
             sum = sum + X(i,:)';
             number_k = number_k + 1;
        end                   
    end
    % calculate the average of the x values, where idk and k match 
    average = sum/number_k;
    % add the average to the kth position of centroids and by that get the new centroid for the certain cluster
    centroids(k,:) = average';
end
    
end