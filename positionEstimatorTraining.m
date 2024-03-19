% This code is submitted my BodyMassIndex:
% Jamie Shing Him Ho
% Mateusz Chodkowski
% Rusne Joneikyte
% Cassius Kua

function [modelParameters] = positionEstimatorTraining(training_data)
% Setup outputs
modelParameters = struct;
analysis_data = struct;

% Setup global parameters
noDirections = size(training_data, 2);
noNeurons = size(training_data(1, 1).spikes, 1);
noTrials = size (training_data, 1);

% Add these variables as parameters
modelParameters.noDirections = noDirections;
modelParameters.noNeurons = noNeurons;

%% PREPROCESSING
% Find neurons with low mutual information - filtering
neurons_to_keep = Entropy_Rusne(training_data,10, 30);
modelParameters.neurons_to_keep = neurons_to_keep;

% Add the new neuron length as parameter and as a variable to use later
noNeurons_filtered = length(neurons_to_keep);
modelParameters.noNeurons_filtered = noNeurons_filtered;

% Change the structure of data to be able to simulate the way the data will
% be fed into the positionEstimator algorithm
% " model will be scored at regular intervals of 20 ms: to start, the first
% 320ms are fed in, then the first 340 ms, then the first 360, 380, 400,
% 420… until the whole recording is eventually fed in to your model" - the
% Competition Document

% organise train data into struct based on incresing window size
i = 1;

 knn_directions = [];
 knn_firing_rates = [];

for dir = 1:noDirections
    for trial = 1:noTrials
        for t = 320:10:560
            mean_firing_rate = [];

            for neuron = 1:noNeurons
                if ismember(neuron, neurons_to_keep)
                    % do mean and then sqrt because that helps
                    update_mean_firing_rate = sqrt(mean(training_data(trial,dir).spikes(neuron,1:t)));
                    mean_firing_rate = [mean_firing_rate update_mean_firing_rate];
                end
            end

            analysis_data(i).mean_firing_rate = mean_firing_rate;
            analysis_data(i).direction = dir;
            analysis_data(i).time = t;
            analysis_data(i).x = training_data(trial,dir).handPos(1,t);
            analysis_data(i).y = training_data(trial,dir).handPos(2,t);
            
            i = i + 1;

            if mod(i, 2) == 0 % take every other trial in order to decrease dimensionality
                knn_directions(end+1) = dir;
                knn_firing_rates = [knn_firing_rates mean_firing_rate'];
            end
        end

    end

end

modelParameters.knn_firing_rates = knn_firing_rates;
modelParameters.knn_directions = knn_directions;

%% KNN

% [features, labels] = preprocessForKNN(training_data);
%
% % Define the range of k values you want to test
% kValues = 1:20;  % Example: testing k from 1 to 20
% numFolds = 5;  % Example: 5-fold cross-validation
%
% % Find the optimal K
% optimalK = findOptimK(features, labels, kValues, numFolds);
% % Store optimalK in modelParameters for later use
% modelParameters.optimalK = optimalK;

best_k = knn_best_k(training_data, 1:noDirections, 5, 10:40);

modelParameters.optimalK = best_k;

%% PCA and LDA for Feature Extraction
data_len = length(analysis_data);

% Setup the arrays
angles = zeros(data_len, 1);
elapsed_times = zeros(data_len, 1);

% PCA & LDa parameters
pcaDimensions = 30; % Number of dimensions to keep after PCA
LDADimensions = 10; % Number of dimensions to keep after LDA

% Prepare data for PCA and LDA
for i = 1:data_len
    angles(i) = analysis_data(i).direction;
    elapsed_times(i) = analysis_data(i).time;
end

% Apply PCA and LDA per direction and time
for a = 1:8
    for t = 1:26
        my_t = t*20 + 300;
        idx = (angles == a) & (elapsed_times == my_t);
        matrix_for_pca = [analysis_data(idx).mean_firing_rate];
        matrix_for_pca = reshape(matrix_for_pca, [], noNeurons_filtered);

        % Applying PCA
        [coeff, score] = my_pca(matrix_for_pca, pcaDimensions);
        anglesVector = [analysis_data(idx).direction];

        % Applying LDA with regularization
        [ldaCoeff, ldaTransformedData,~,~] = mylda_adjusted(score, anglesVector, pcaDimensions, LDADimensions);

        modelParameters.PCA(a,t).coeff = coeff;
        modelParameters.LDA(a,t).coeff = ldaCoeff;

        % Store the LDA transformed data for regression
        modelParameters.LDA(a,t).transformed_data = ldaTransformedData;

        % Store x and y coordinates for regression
        modelParameters.regression_data(a,t).x = [analysis_data(idx).x]';
        modelParameters.regression_data(a,t).y = [analysis_data(idx).y]';
    end
end

%% Linear Regression Models for X and Y coordinates
lambda = 0.001; % Example lambda value for L1 regularization; adjust based on validation
iterations = 100; % Number of iterations for the iterative process
learningRate = 0.001; % Learning rate for gradient step

for a = 1:8
    for t = 1:26
        transformed_data = modelParameters.LDA(a,t).transformed_data;
        x = modelParameters.regression_data(a,t).x;
        y = modelParameters.regression_data(a,t).y;

        % Prepare the design matrix with an intercept term
        X_design_x = [ones(size(transformed_data, 1), 1), transformed_data];
        X_design_y = X_design_x; % Identical design matrix for Y

        % Initialize coefficients for x and y
        coef_x = zeros(size(X_design_x, 2), 1);
        coef_y = zeros(size(X_design_y, 2), 1);

        % Iterative L1 regularization (Lasso) for X
        for iter = 1:iterations
            % Gradient step
            y_pred = X_design_x * coef_x;
            grad = X_design_x' * (y_pred - x) + lambda * sign(coef_x);
            coef_x = coef_x - learningRate * grad;

            % Soft thresholding for each coefficient except the intercept
            coef_x(2:end) = sign(coef_x(2:end)) .* max(abs(coef_x(2:end)) - lambda * learningRate, 0);
        end

        % Iterative L1 regularization (Lasso) for Y
        for iter = 1:iterations
            % Gradient step
            y_pred = X_design_y * coef_y;
            grad = X_design_y' * (y_pred - y) + lambda * sign(coef_y);
            coef_y = coef_y - learningRate * grad;

            % Soft thresholding for each coefficient except the intercept
            coef_y(2:end) = sign(coef_y(2:end)) .* max(abs(coef_y(2:end)) - lambda * learningRate, 0);
        end

        modelParameters.regression(a,t).coef_x = coef_x;
        modelParameters.regression(a,t).coef_y = coef_y;
    end
end


%% FUNCTIONS
    function [coefficients, scores] = my_pca(X, num_components)
        % Standardize the dataset
        X = X - mean(X);
        X = X ./ std(X);

        % Compute covariance matrix and perform eigen decomposition
        C = cov(X);
        [V, D] = eig(C);

        % Sort eigenvectors by eigenvalues in descending order
        [~, order] = sort(diag(D), 'descend');
        V = V(:, order);

        % Select the leading eigenvectors
        coefficients = V(:, 1:num_components);
        scores = X * coefficients;
    end


%% lda function
    function [W, OptimalOutput, EigenVectorsLDA, EigenValuesLDA] = mylda_adjusted(pcaScores, classLabels, PCADimensions, LDADimensions)
    % Ensure classLabels are column vectors
    if isrow(classLabels)
        classLabels = classLabels';
    end

    % Reduce PCA scores to the specified number of dimensions
    pcaScoresReduced = pcaScores(:, 1:PCADimensions);

    % Calculate the mean of the whole dataset
    overallMean = mean(pcaScoresReduced, 1);

    % Initialize within-class scatter matrix Sw and between-class scatter matrix Sb
    Sw = zeros(size(pcaScoresReduced, 2), size(pcaScoresReduced, 2));
    Sb = zeros(size(pcaScoresReduced, 2), size(pcaScoresReduced, 2));

    uniqueClasses = unique(classLabels);
    numClasses = length(uniqueClasses);

    % Calculate Sw and Sb
    for i = 1:numClasses
        % Indices of rows for the current class
        classIndices = classLabels == uniqueClasses(i);

        % Scores for the current class
        classScores = pcaScoresReduced(classIndices, :);

        % Mean of the current class
        classMean = mean(classScores, 1);

        % Calculate within-class scatter matrix
        Sw = Sw + (classScores - classMean)' * (classScores - classMean);

        % Calculate between-class scatter matrix
        classDiff = classMean - overallMean;
        Sb = Sb + length(classIndices) * (classDiff' * classDiff);
    end

    % Solve the generalized eigenvalue problem for Sb and Sw
    [eigenvectors, eigenvalues] = eig(Sb, Sw);

    % Sort eigenvalues and eigenvectors
    [sortedEigenvalues, sortIndex] = sort(diag(eigenvalues), 'descend');
    sortedEigenvectors = eigenvectors(:, sortIndex);

    % Select the LDADimensions most discriminative directions
    W = sortedEigenvectors(:, 1:LDADimensions);
    OptimalOutput = pcaScoresReduced * W;

    EigenVectorsLDA = W;
    EigenValuesLDA = diag(sortedEigenvalues(1:LDADimensions));
end
%% optimal KNN - Not used, as requires toolboxes

    % function [features, labels] = preprocessForKNN(training_data)
    %     % Initialize arrays to hold features and labels
    %     features = [];
    %     labels = [];
    % 
    %     for trialIdx = 1:size(training_data, 1)
    %         for conditionIdx = 1:size(training_data, 2)
    %             % Assuming each condition has multiple observations
    %             % Extract features for each observation in the trial and condition
    %             % This part depends on the structure of your data
    %             trialFeatures = extractFeaturesFromTrial(training_data(trialIdx, conditionIdx));
    %             trialLabels = repmat(conditionIdx, size(trialFeatures, 1), 1);
    % 
    %             % Append to overall feature and label arrays
    %             features = [features; trialFeatures];
    %             labels = [labels; trialLabels];
    %         end
    %     end
    % end
    % 
    % function optimalK = findOptimalK(features, labels, kValues, numFolds)
    %     cvIndices = crossvalind('Kfold', labels, numFolds);
    %     meanAccuracy = zeros(length(kValues), 1);
    % 
    %     for kIndex = 1:length(kValues)
    %         k = kValues(kIndex);
    %         foldAccuracy = zeros(numFolds, 1);
    % 
    %         for fold = 1:numFolds
    %             testIdx = (cvIndices == fold);
    %             trainIdx = ~testIdx;
    % 
    %             % Directly use the indexed features and labels for the current fold
    %             XTrain = features(trainIdx, :);
    %             yTrain = labels(trainIdx);
    %             XTest = features(testIdx, :);
    %             yTest = labels(testIdx);
    % 
    %             % Train KNN and evaluate accuracy for the current fold
    %             Mdl = fitcknn(XTrain, yTrain, 'NumNeighbors', k);
    %             yPred = predict(Mdl, XTest);
    % 
    %             % Assuming you have a function calculateAccuracy to evaluate your model
    %             % You'll need to define this based on your accuracy measurement
    %             foldAccuracy(fold) = calculateAccuracy(yPred, yTest);
    %         end
    % 
    %         meanAccuracy(kIndex) = mean(foldAccuracy);
    %     end
    % 
    %     [~, optimalKIdx] = max(meanAccuracy);
    %     optimalK = kValues(optimalKIdx);
    % end
    % 
    % function accuracy = calculateAccuracy(yPred, yTrue)
    %     if isempty(yPred) || isempty(yTrue) || length(yPred) ~= length(yTrue)
    %         error('Inputs must be non-empty and of equal length.');
    %     end
    % 
    %     % Calculate the number of correct predictions
    %     correctPredictions = sum(yPred == yTrue);
    % 
    %     % Calculate accuracy
    %     accuracy = correctPredictions / length(yTrue);
    % end
    % 
    % 
    % function trialFeatures = extractFeaturesFromTrial(trial)
    %     meanFiringRates = mean(trial.spikes, 2); % Mean across time, for each neuron
    % 
    %     trialFeatures = meanFiringRates; % If using only mean firing rates as features
    % 
    %     % Transpose if necessary to ensure trialFeatures is a row vector
    %     if iscolumn(trialFeatures)
    %         trialFeatures = trialFeatures';
    %     end
    % end

%% Preprocessing functions

    function neuron_high_MI = Entropy_Rusne(test_data, bin_size, prc)

        % trim and bin data
        data = trimAndBinToSum(test_data,300, 500, bin_size);

        % extract the dimensions
        noTrials = data.noTrials;
        noDirections = data.noDirections;
        noNeurons = data.noNeurons;

        spike_count_r = zeros(noNeurons, data.maxBinSum+1);
        spike_count_r_given_s = zeros(noNeurons, noDirections, data.maxBinSum+1);

        edges = -0.5: 1: data.maxBinSum + 0.5;

        for n = 1:noNeurons
            for j=1:noDirections
                for i = 1:noTrials
                    update = histcounts(data.trial_data(i, j).bin_sums(n, :), edges); %count how many times each value has occured, return array of the frequency for each value
                    spike_count_r(n, :) = spike_count_r(n, :) + update; % add to previous list to get the total number a value occured in all trials, all directions, for each neuron
                    spike_count_r_given_s(n, j, :) = spike_count_r_given_s(n, j, :) + permute(update, [3, 1, 2]); % add to spike count, but separate for each direction and each neuron
                end
            end
        end

        total_r_per_neuron = sum(spike_count_r, 2); % divide each spike count by the number of spikes of that bin in all trials, all directions but a single neuron
        total_r_per_neuron_per_state = sum(spike_count_r_given_s, 3);

        P_r = spike_count_r ./total_r_per_neuron;
        P_r_given_s = spike_count_r_given_s ./ permute(total_r_per_neuron_per_state, [1, 2, 3]);
        P_s = 1/noDirections;

        H_r = -P_r .* log2(P_r + (P_r == 0));
        H_r_given_s = -P_s .*P_r_given_s .* log2(P_r_given_s + (P_r_given_s == 0));

        H_R = sum (H_r, 2);
        H_R_given_s = sum (H_r_given_s, 3);
        H_R_given_S = sum (H_R_given_s, 2);

        MI = H_R - H_R_given_S;

        % median_MI = median (MI);
        % neuron_high_MI = find (MI > median_MI);

        median_MI = prctile (MI, prc);
        neuron_high_MI = find (MI > median_MI);

    end

% Another entropy function is used in the code

    % function neuron_high_MI = Entropy_Mati(test_data, bin_size)
    % 
    %     noTrials = size(test_data,1);
    %     noDirections = size(test_data,2);
    %     noNeurons = size(test_data(1,1).spikes,1);
    % 
    %     firing_rate_neurons = zeros(noNeurons, noDirections, noTrials);
    %     % bin_size = 10;
    %     noBins = 300/bin_size;
    % 
    %     for ang = 1:noDirections
    %         for t = 1:noTrials
    %             for neuron = 1:noNeurons
    %                 spike_train_binned = zeros(noBins,1);
    %                 spike_train = test_data(t,ang).spikes(neuron,200:499)';
    % 
    %                 for bin = 1:noBins
    %                     spike_train_binned(bin) = sum(spike_train(1+bin_size*(bin-1):bin_size*bin));
    %                 end
    %                 % spike_train_binned = gaussian_Rusne(spike_train_binned);
    %                 firing_rate_neurons(neuron, ang, t) = mean(spike_train_binned);
    %             end
    %         end
    %         firing_rate_stims = mean(firing_rate_neurons, 3);
    %     end
    %     firing_rate = mean(firing_rate_stims, 2);
    % 
    %     p_stim = 1/noDirections;
    % 
    %     Mi = zeros(noNeurons, 1);
    % 
    %     for px = 1:noNeurons
    %         for py = 1:noDirections
    %             Mi(px) = Mi(px) + firing_rate_stims(px,py) * log(firing_rate_stims(px,py) / (firing_rate(px) * p_stim));
    %         end
    %     end
    % 
    % 
    %     Mi_index = [[1:98]', Mi];
    % 
    %     Mi_index(any(isnan(Mi_index), 2), :) = [];
    %     Mi_sort = sortrows(Mi_index, 2, 'descend');
    % 
    %     th = 10;
    % 
    %     Mi_th = sortrows(Mi_sort(1:th,:), 1);
    % 
    %     neuron_high_MI = Mi_th(:,1);
    % 
    % end

% Not used, as the functions use toolboxes:

    % function smoothedData = gaussian_Rusne(binnedData)
    % 
    %     % Standard deviation of the Gaussian kernel (adjust as needed)
    %     sigma = 2; %TODO: test different sigma
    % 
    %     % Create the Gaussian kernel
    %     kernelSize = 5 * sigma; % Choose an appropriate kernel size
    %     kernel = fspecial('gaussian', [1, kernelSize], sigma);
    % 
    %     % Preallocate smoothed data matrix
    %     smoothedData = zeros(size(binnedData));
    % 
    %     % Apply Gaussian smoothing to each neuron's spike train data
    %     smoothedData(:) = conv(binnedData(:), kernel, 'same');
    % 
    % end
    % 
    % 
    % function smoothedData = gaussian_Rusne_whole_data_set(binnedData)
    %     % Standard deviation of the Gaussian kernel (adjust as needed)
    %     sigma = 2;
    % 
    %     % Create the Gaussian kernel
    %     kernelSize = 5 * sigma; % Choose an appropriate kernel size
    %     kernel = fspecial('gaussian', [1, kernelSize], sigma);
    % 
    %     % Preallocate smoothed data matrix
    %     smoothedData = zeros(size(binnedData));
    % 
    %     % Apply Gaussian smoothing to each neuron's spike train data
    %     for i = 1:size(binnedData, 1) % Iterate over each neuron
    %         smoothedData(i, :) = conv(binnedData(i, :), kernel, 'same');
    %     end
    % 
    % end


    function trial_Processed = trimAndBinToSum(raw_Data, start_time, end_time, bin_Size)
        % Takes a struct of the input data
        % Trims it to length that is the most relative - start_time to end_time
        % Bins the spikes of each trial by summing the number of spikes in a bin_Size interval
        % The dimensions are included in the struct

        % Set up the struct and other variables
        trial_Processed = struct;

        % Number of:
        noTrials = size(raw_Data,1);
        noDirections = size(raw_Data,2);
        noNeurons = size(raw_Data(1,1).spikes,1);
        noBins = floor((end_time - start_time)/bin_Size);

        % Set up the dimensions to be in the struct
        trial_Processed.bin_Size = bin_Size;
        trial_Processed.noDirections = noDirections;
        trial_Processed.noTrials = noTrials;
        trial_Processed.noNeurons = noNeurons;

        for i=1:noTrials

            for j = 1:noDirections

                % trim data for all neurons
                data = raw_Data(i, j).spikes(:, start_time : end_time - 1);

                for k = 1:noBins
                    bin_sum(:,k) = sum(data(:,(k-1)*(bin_Size)+1:(k*bin_Size)), 2);
                end
                trial_Processed.trial_max(i,j) = max(max(bin_sum));
                trial_Processed.trial_data(i,j).bin_sums = bin_sum;
            end

            trial_Processed.maxBinSum = max(max(trial_Processed.trial_max));
        end
    end

%% Toolboxless KNN
    function k_best = knn_best_k(data, stimuli, folds, k_candidates)

        labels = repmat(stimuli', size(data, 1), 1);

        firing_rates = [];

        for t = 1:size(data, 1)
            for ang = 1:length(stimuli)
                firing_rates(end + 1, :) = mean(data(t, ang).spikes, 2);
            end 
        end 

        cv_indeces_ordered = repmat([1:folds]', (size(data, 1) * length(stimuli)) / 5, 1);
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
end