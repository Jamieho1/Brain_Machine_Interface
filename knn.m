function k_best = knn_best_k(firing_rates, stimuli, folds, k_candidates)

labels = repmat(stimuli', size(firing_rates, 1), 1);

cv_indeces_ordered = repmat([1:folds]', (size(firing_rates, 1) * length(stimuli)) / 5, 1);
cv_indices = cv_indeces_ordered(randperm(length(cv_indeces_ordered)));

k_accuracy_score = zeros(length(k_candidates), 1);

for k = k_candidates

    fold_acc = zeros(folds, 1);
    for f = 1:folds
        test = (cv_indices == f); 
        train = ~test;

        xtrain = firing_rates(train, :);
        ytrain = labels(train);
        xtest = firing_rates(test, :);
        ytest = labels(test);

        y_prediction = KNN(xtrain, ytrain, xtest, k);

        accuracy_score =  sum(y_prediction == ytest) / length(ytest);  
        fold_acc(f) = accuracy_score;

    end 
    k_accuracy_score(k) = mean(fold_acc);
end 

[~,k_best] = max(k_accuracy_score);  

end 



function prediction = KNN(xtrain, ytrain, xtest, k)

    prediction = zeros(size(xtest, 1), 1);

    for xxtest = 1:size(xtest, 1)
        distances = sqrt(sum((xtrain - xtest(xxtest, :)).^2, 2));
        [~, indices_sorted] = sort(distances, 'ascend');
        top_k_indices = indices_sorted(1:k);
        prediction(xxtest) = mode(ytrain(top_k_indices));
    end 

end 