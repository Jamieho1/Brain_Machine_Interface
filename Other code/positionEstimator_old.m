function [x, y] = positionEstimator_old(test_data, modelParameters)
%% KNN

neurons_len = modelParameters.neurons_len;
mean_firing_rates = zeros(1, neurons_len);
to_remove = modelParameters.to_remove;
cc=0;
x = 400; % set the threshold value for the number of spikes

% Calculate mean firing rates
for n = 1:98
    if ~ismember(n, to_remove)
        cc = cc + 1;
        num_spikes = length(test_data.spikes(n,:));
        if num_spikes > x
            time_interval = x;
            spike_subset = test_data.spikes(n, 1:x);
            mean_firing_rates(cc) = sqrt(mean(spike_subset));
        else
            time_interval = num_spikes;
            mean_firing_rates(cc) = sqrt(mean(test_data.spikes(n,:)));
        end
    end
end

%% Regression
% Predict angle using KNN
test_angle = KNN(modelParameters.mean_rates, mean_firing_rates, modelParameters.optimalK, time_interval);

% Calculate 't' based on test_data.spikes
elapsed_time = size(test_data.spikes, 2);
    t = min(floor(elapsed_time / 20) -15, 13);

% PCA and LDA transformations
pca_coeff = modelParameters.PCA(test_angle, t).coeff;
lda_coeff = modelParameters.LDA(test_angle, t).coeff;
pca_transformed = mean_firing_rates * pca_coeff;
lda_transformed = pca_transformed * lda_coeff;

% Predict x and y
x = [1, lda_transformed] * modelParameters.regression(test_angle,t).coef_x;
y = [1, lda_transformed] * modelParameters.regression(test_angle,t).coef_y;
end

%% FUNCTIONS 

% Coded with reference to 
% https://towardsdatascience.com/k-nearest-neighbours-introduction-to-machine-learning-algorithms-18e7ce3d802a
function [label] = KNN(final_train_data,final_test,k1, time_interval)
   
   idx = [final_train_data.elapsed_time] >= (time_interval - 40) & [final_train_data.elapsed_time] <= (time_interval+20);
   final_train_data = final_train_data(idx);
    
   m = length(final_train_data); 
    
    distances = zeros(m, 1);
    for i = 1:m
        % Only calculate distance if elapsed_time matches time_interval
            d = norm(final_train_data(i).mean_activity - final_test);
            distances(i) = d;
    end

    % Sort the distances in ascending order and keep track of the indices
    [~, indices] = sort(distances);

    % Find the k-nearest neighbors
    kNearestNeighbors = final_train_data(indices(1:k1));

    % Count the number of occurrences of each label in the k-nearest neighbors
    labelCounts = zeros(1, 8);
    for i = 1:k1
        labelCounts(kNearestNeighbors(i).angle) = labelCounts(kNearestNeighbors(i).angle) + 1;
    end

    [~, label] = max(labelCounts);
    
end

