% MULTI-CLASS CLASSIFICATIONS - ONE-VS-ALL PREDICTION

function p = predictOneVsAll(all_theta, X)
% The function will return a vector of predictions for each example in the matrix X. Note that X contains the examples in rows. 
% 'all_theta' is a matrix where the i-th row is a trained logistic regression theta vector for the i-th class. 


m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];  

prediction = sigmoid(X * all_theta');

[m, p]=max(prediction, [], 2);


end