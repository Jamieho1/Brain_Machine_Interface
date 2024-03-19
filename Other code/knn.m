function [k_best, k_best_acc] = knn_best_k(data, stimuli, folds, k_candidates)

labels = repmat(stimuli', size(data, 1), 1);

firing_rate_neurons = [];

for t = 1:size(data, 1) 
    for ang = 1:length(stimuli)
            firing_rate_neurons(end + 1, :) = mean(data(t,ang).spikes, 2);
    end 
end 

cv_indices = crossvalind('Kfold', labels, folds);
k_accuracy_score = zeros(length(k_candidates), 1);

for k = k_candidates
    fold_acc = zeros(folds, 1);
    for f = 1:folds
        test = (cv_indices == f); 
        train = ~test;

        xtrain = firing_rate_neurons(train, :);
        ytrain = labels(train);
        xtest = firing_rate_neurons(test, :);
        ytest = labels(test);

        knn_model = fitcknn(xtrain, ytrain, 'NumNeighbors', k);
        yprediction = predict(knn_model, xtest);

        accuracy_score =  sum(yprediction == ytest) / length(ytest);  
        fold_acc(f) = accuracy_score;

    end 
    k_accuracy_score(k) = mean(fold_acc);
end 

[k_best_acc,k_best] = max(k_accuracy_score);  

end 