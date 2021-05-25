%%%% ANOMALY DETECTION - SELECT EPSILON %%%%

function [bestEpsilon bestF1] = selectThreshold(yval, pval)
% finds the best threshold to use for selecting outliers based on the results from a validation set (pval) and the ground truth (yval).

% Initialize variables
bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % Get et a binary vector of 0's and 1's of the outlier predictions
    pred = (pval < epsilon);
    
    % determine true positive (tp), false positive (fp) and false negative (fn)
    % tp when your algorithm considers x-CV(i) an anomaly (=1) and x-CV(i) is labeled as anomalous
    tp = sum((pred == 1) & (yval == 1));
    % fp when your algorithm considers x-CV(i) an anomaly and x-CV(i) is labeled as normal (=0)
    fp = sum((pred == 1) & (yval == 0));
    % fn when your algorithm considers x-CV(i) normal and x-CV(i) is labeled as anomalous
    fn = sum((pred == 0) & (yval == 1));
    
    % calculate precision (prec) and recall (rec)
    prec = tp/(tp + fp);
    rec = tp/(tp + fn);
    
    % calculate F1 score
    F1 = (2 * prec * rec)/(prec + rec);
    
    % if an F1 is found that is higher than bestF1 -> set as new F1 and set that epsilon as bestEpsilon
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
