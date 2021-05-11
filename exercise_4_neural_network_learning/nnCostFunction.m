% NEURAL NETWORK LEARNING - COST FUNCTION

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% The function computes the cost and gradient of the neural network. 
% The parameters for the neural network are "unrolled" into the vector nn_params and need to be converted back into the weight matrices.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Add ones to the X data matrix
X = [ones(m, 1) X];

% MAP VECTOR y INTO A BINARY VECTOR OF 1's AND 0's
% Vector y to Matrix Y
Y = zeros(m, num_labels);
% Loop through each row
for i = 1:m
    % Use the value of y as an index; set the value matching index to 1
    Y(i, y(i)) = 1;
end

% FORWARD PROPAGATION
%hidden layer in layer 2
z2 = X * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
%output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% COST FUNCTION WITHOUT REGULARIZATION
J = (-1/m)*(sum(sum(Y .* log(a3) + (1 - Y) .* log(1 - a3))));

% COST FUNCTION WITH REGULARIZATION 
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).^2,2)));

%BACKPROPAGATION

for t = 1:m
   Y = zeros(num_labels, m);
   % Loop through each row
   for i = 1:m
    % Use the value of y as an index; set the value matching index to 1
    Y(y(i), i) = 1;
   end
   
   %FORWARD PROPAGATION
   a1 = X(t,:);
   z2 = Theta1 * a1';
   a2 = [1; sigmoid(z2)];
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);
   
   %BACKPROPAGATION
   z2_1 = [1; z2];
   delta_3 =  a3 - Y(:,t);
   delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2_1);
   
   % skip or remove delta_2(0)
   delta_2 = delta_2(2:end); 
   
   % Accumulating gradient
   Theta2_grad = Theta2_grad + (delta_3 * a2');
   Theta1_grad = Theta1_grad + (delta_2 * a1);    

end;

% REGULARIZED GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) /m + (lambda/m * Theta1(:,2:end));
Theta1_grad(:,1) = Theta1_grad(:,1) / m;

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) / m + (lambda/m * Theta2(:,2:end));
Theta2_grad(:,1) = Theta2_grad(:,1) / m;
    

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end