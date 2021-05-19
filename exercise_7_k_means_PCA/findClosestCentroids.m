% K-MEANS - CLOSEST CENTROID INDEX


function idx = findClosestCentroids(X, centroids)
% returns the closest centroids index in idx for a dataset X where each row is a single example. 
% idx = m x 1 vector of centroid assignments (i.e. each entry in range [1..K])

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = length(X);

%loop through examples
for i = 1:m
    dist = Inf;
    best_cent = 0;

    % loop through K (# of centroids)
    for j = 1:K
        % calculate distance
        v = X(i,:) - centroids(j,:);
        dist_new = sum(v * v');

        % if the new distance is smaller than the old one, save new min distance and the index j    
        if dist_new < dist
            dist = dist_new;
            best_cent = j;
        end
    end

    % add the index of the centroid closest to x(i) to idx 
    idx(i) = best_cent;
end

end