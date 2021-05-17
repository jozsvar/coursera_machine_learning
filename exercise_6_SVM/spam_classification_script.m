% SPAM CLASSIFICATION - SCRIPT 

% Extract Features from text file
file_contents = readFile('emailSample1.txt');
% go through eMail, match words with vocabulary list and add matched words into word_indices 
word_indices  = processEmail(file_contents);

% Print Stats
disp(word_indices)

% Extract features from eMail: creates feature vector with 0 and 1 for each word that appears in word_indices
features = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

% Training SVM for spam classification
% Load the Spam Email dataset: X and y in the environment
% Dataset contains 4000 training examples of spam and non-spam email
load('spamTrain.mat');

C = 0.1;
%train a SVM to classify between spam (y = 1) and non-spam (y = 0) emails
model = svmTrain(X, y, C, @linearKernel);

% determine the training accuracy
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% Load the test dataset: Xtest, ytest in your environment
% Dataset contains 1000 test examples
load('spamTest.mat');

% determine the test accuracy 
p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

% Top predictors for spam
% Sort the weights and obtain the vocabulary list
[weight, idx] = sort(model.w, 'descend');

% The code finds the parameters with the largest positive values in the classier and displays the corresponding words
vocabList = getVocabList();
for i = 1:15
    if i == 1
        fprintf('Top predictors of spam: \n');
    end
    fprintf('%-15s (%f) \n', vocabList{idx(i)}, weight(i));
end;



