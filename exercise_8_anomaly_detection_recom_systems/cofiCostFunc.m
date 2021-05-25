%%%% RECOMMENDER SYSTEMS - COST FUNCTION %%%%

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

% returns the cost and gradient for the collaborative filtering problem.

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);

Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% COST FUNCTION WITHOUT REGULARIZATION %
cost = (X * Theta' - Y) .* R;
% accumulate the cost for user j and movie i only if R(i,j) = 1
J = (1/2) * sum(sum(cost.^2));  

% GRADIENT DESCENT WITHOUT REGULARIZATION % 
X_grad = cost * Theta;
Theta_grad = cost' * X;

% COST FUNCTION REGULARIZED %
reg_theta = (lambda/2) * sum(sum(Theta.^2));
reg_X = (lambda/2) * sum(sum(X.^2));
J = J + reg_theta + reg_X;

% GRADIENT DESCENT REGULARIZATION %
X_grad = X_grad + (lambda * X);
Theta_grad = Theta_grad + (lambda * Theta);


grad = [X_grad(:); Theta_grad(:)];

end
