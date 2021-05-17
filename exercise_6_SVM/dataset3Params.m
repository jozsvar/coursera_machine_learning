% SUPPORT VECTOR MACHINE - DETERMINE OPTIMAL C AND SIGMA PARAMETERS

function [C, sigma] = dataset3Params(X, y, Xval, yval)
% returns your choice of C and sigma. 
% returns the optimal C and sigma based on a cross-validation set.


% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% chosen values
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% set initial error
error = Inf;

for c_i = 1:length(values)
    for sigma_i = 1:length(values)
        % train an SVM classifier and return trained model + gaussian kernel
        model = svmTrain(X, y, values(c_i), @(x1, x2) gaussianKernel(x1, x2, values(sigma_i)));
        % predict the labels on the cross validation set
        pred = svmPredict(model, Xval);
        % compute the prediction error
        pred_error = mean(double(pred ~= yval));
        fprintf('C = %.2f, sigma = %.2f, error =  %.2f', values(c_i), values(sigma_i), pred_error);
        
        % determine if new minimum error was found and use those C, sigma and error for next loop 
        if pred_error < error
            C = values(c_i);
            sigma = values(sigma_i);
            error = pred_error;
        end
    end
end

end
