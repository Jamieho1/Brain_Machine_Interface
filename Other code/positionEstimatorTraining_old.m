function [modelParameters] = positionEstimatorTraining_old(training_data)
n_conditions = 8;
final_train_data = struct;
modelParameters = struct;
train_data_len = length(training_data);
knn_data = struct;

%% Preprocessing
% Find neurons with low mutual information - filtering
neuron_list_high_MI = Entropy_Rusne(training_data,10, 30); % TODO test different bin sizes
modelParameters.neuron_list_high_MI = neuron_list_high_MI;

% organise train data into struct based on incresing window size 
count = 1;
c = 1;
for cond = 1:n_conditions
    for i = 1:train_data_len
        
         times = 320:20:600;
       
        for l = times 
            mean_activity = [];
            for neuron = 1:98
                if ismember(neuron, neuron_list_high_MI)
                    % do mean and then sqrt because that helps
                    neuron_mean_activity = sqrt(mean(training_data(i,cond).spikes(neuron,1:l)));
                    mean_activity = [mean_activity neuron_mean_activity]; 
                end
            end
            
            if l <= 400 && mod(i,2)
                knn_data(c).angle = cond;
                knn_data(c).elapsed_time = l;
                knn_data(c).mean_activity = mean_activity;
                c = c + 1;
            end
        
            final_train_data(count).elapsed_time = l;
            final_train_data(count).mean_activity = mean_activity;
            final_train_data(count).angle = cond;
            final_train_data(count).x = training_data(i,cond).handPos(1,l);
            final_train_data(count).y = training_data(i,cond).handPos(2,l);
            count = count + 1;
        end
      
    end
        
end

%% KNN

[features, labels] = preprocessForKNN(training_data);
    
% Define the range of k values you want to test
kValues = 1:20;  % Example: testing k from 1 to 20
numFolds = 5;  % Example: 5-fold cross-validation 

% Find the optimal K
optimalK = findOptimalK(features, labels, kValues, numFolds);
% Store optimalK in modelParameters for later use
modelParameters.optimalK = optimalK;

%% PCA and LDA for Feature Extraction
modelParameters.mean_rates = knn_data;
neurons_len = length(neuron_list_high_MI);
modelParameters.neurons_len = neurons_len;
data_len = length(final_train_data);
angles = zeros(data_len, 1);
elapsed_times = zeros(data_len, 1);
lambda = 0.01;
pcaDimensions = 10; % Number of dimensions to keep after PCA


% Prepare data for PCA and LDA
for i = 1:data_len
    angles(i) = final_train_data(i).angle;
    elapsed_times(i) = final_train_data(i).elapsed_time;
end

% Apply PCA and LDA per angle and time
for a = 1:8
    for t = 1:1
        my_t = t*20 + 300;
        idx = (angles == a) & (elapsed_times == my_t);
        pca_matrix = [final_train_data(idx).mean_activity];
        pca_matrix = reshape(pca_matrix, [], neurons_len);

        % Applying PCA
        [coeff, score] = my_pca(pca_matrix, pcaDimensions);
        anglesVector = [final_train_data(idx).angle];

        % Applying LDA with regularization
        ldaScores = mylda(score, anglesVector, lambda);

        modelParameters.PCA(a,t).coeff = coeff;
        modelParameters.LDA(a,t).coeff = ldaScores.coeff;

        % Store the LDA transformed data for regression
        modelParameters.LDA(a,t).transformed_data = ldaScores.transformed_data;

        % Store x and y coordinates for regression
        modelParameters.regression_data(a,t).x = [final_train_data(idx).x]';
        modelParameters.regression_data(a,t).y = [final_train_data(idx).y]';
    end
end

%% Linear Regression Models for X and Y coordinates
for a = 1:8
    for t = 1:1
        transformed_data = modelParameters.LDA(a,t).transformed_data;
        x = modelParameters.regression_data(a,t).x;
        y = modelParameters.regression_data(a,t).y;

        modelParameters.regression(a,t).coef_x = regress(x, [ones(size(transformed_data, 1), 1), transformed_data]);
        modelParameters.regression(a,t).coef_y = regress(y, [ones(size(transformed_data, 1), 1), transformed_data]);
    end
end


%% FUNCTIONS
% Coded with reference to 
% https://www.youtube.com/watch?v=FgakZw6K1QQ
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

function ldaScores = mylda(pcaScores, classLabels, lambda)

    if nargin < 3
        lambda = 0.01; % Default regularization parameter if not provided
    end

    uniqueClasses = unique(classLabels);
    numClasses = length(uniqueClasses);
    overallMean = mean(pcaScores, 1);
    Sb = zeros(size(pcaScores, 2));
    Sw = zeros(size(pcaScores, 2)) + lambda * eye(size(pcaScores, 2)); % Adding regularization
    
    for i = 1:numClasses
        classScores = pcaScores(classLabels == uniqueClasses(i), :);
        classMean = mean(classScores, 1);
        Sb = Sb + size(classScores, 1) * (classMean - overallMean)' * (classMean - overallMean);
        Sw = Sw + (classScores - classMean)' * (classScores - classMean);
    end

    [eigenvectors, ~] = eig(Sb, Sw); % Directly using generalized eigenvalue problem solver
    ldaTransformedData = pcaScores * eigenvectors;

    ldaScores.coeff = eigenvectors;
    ldaScores.transformed_data = ldaTransformedData;
end

%% optimal K KNN 

function [features, labels] = preprocessForKNN(training_data)
    % Initialize arrays to hold features and labels
    features = [];
    labels = [];
    
    for trialIdx = 1:size(training_data, 1)
        for conditionIdx = 1:size(training_data, 2)
            % Assuming each condition has multiple observations
            % Extract features for each observation in the trial and condition
            % This part depends on the structure of your data
            trialFeatures = extractFeaturesFromTrial(training_data(trialIdx, conditionIdx));
            trialLabels = repmat(conditionIdx, size(trialFeatures, 1), 1);
            
            % Append to overall feature and label arrays
            features = [features; trialFeatures];
            labels = [labels; trialLabels];
        end
    end
end

function optimalK = findOptimalK(features, labels, kValues, numFolds)
    cvIndices = crossvalind('Kfold', labels, numFolds);
    meanAccuracy = zeros(length(kValues), 1);
    
    for kIndex = 1:length(kValues)
        k = kValues(kIndex);
        foldAccuracy = zeros(numFolds, 1);
        
        for fold = 1:numFolds
            testIdx = (cvIndices == fold);
            trainIdx = ~testIdx;
            
            % Directly use the indexed features and labels for the current fold
            XTrain = features(trainIdx, :);
            yTrain = labels(trainIdx);
            XTest = features(testIdx, :);
            yTest = labels(testIdx);
            
            % Train KNN and evaluate accuracy for the current fold
            Mdl = fitcknn(XTrain, yTrain, 'NumNeighbors', k);
            yPred = predict(Mdl, XTest);
            
            % Assuming you have a function calculateAccuracy to evaluate your model
            % You'll need to define this based on your accuracy measurement
            foldAccuracy(fold) = calculateAccuracy(yPred, yTest);
        end
        
        meanAccuracy(kIndex) = mean(foldAccuracy);
    end
    
    [~, optimalKIdx] = max(meanAccuracy);
    optimalK = kValues(optimalKIdx);
end

function accuracy = calculateAccuracy(yPred, yTrue)
    if isempty(yPred) || isempty(yTrue) || length(yPred) ~= length(yTrue)
        error('Inputs must be non-empty and of equal length.');
    end
    
    % Calculate the number of correct predictions
    correctPredictions = sum(yPred == yTrue);
    
    % Calculate accuracy
    accuracy = correctPredictions / length(yTrue);
end


function trialFeatures = extractFeaturesFromTrial(trial)
    meanFiringRates = mean(trial.spikes, 2); % Mean across time, for each neuron
    
    trialFeatures = meanFiringRates; % If using only mean firing rates as features
    
    % Transpose if necessary to ensure trialFeatures is a row vector
    if iscolumn(trialFeatures)
        trialFeatures = trialFeatures';
    end
end


%% preprocessing function
function removed = removeVariance(training_data,threshold_percentage)
% Calculate the variance across 8 directions
% If does not meet threshold (threhold_percentage), remove the neuron.
% Input rates should be 3D arraw of neuron vs time vs trial
% Compare variability across trials of same direction to variability across
% each direction.
% Returns a vector containing the neuron numbers that should be removed.
% Calculate variance across trials with same stimulus. Returns a 98x1
% vector of variances

bin_Size = 10;
gauss_Window = 50;
rates = bintoRates(training_data,bin_Size,1);
rates = gaussianKernel(rates,20,gauss_Window,1);
end_Time = rates.end_Time;
rates = rates.rates;
numNeurons = 98;
numDirections = 8;
numTrials = size(rates,1);

% holds the variance for each neuron across all training data for each
% respective direction.
var_intrial = zeros(numNeurons,numDirections);
% go along each trial for each direction
for direction = 1:numDirections
    rates_intrial = zeros(numNeurons,end_Time*numTrials);
    i = 1;
    for trials = 1:numTrials
            rates_intrial(:,(i-1)*end_Time+1:i*end_Time)=rates(trials,direction).rates(:,1:end_Time);           
            i = i+1;
    end
    var_intrial(:,direction) = var(rates_intrial,0,2);
end
mean_varintrial = mean(var_intrial,2);

% holds variance for each neuron across all directions 
var_directions = zeros(numNeurons,numTrials);
for trials = 1:numTrials
    rates_directions = zeros(numNeurons, end_Time*numDirections);
    i = 1;
    for direction = 1:numDirections
            rates_directions(:,(i-1)*end_Time+1:i*end_Time) = rates(trials,direction).rates(:,1:end_Time);         
            i = i+1;
    end
    var_directions(:,trials) = var(rates_directions,0,2);
end
mean_vardirections = mean(var_directions,2);

percentagediff = ((mean_vardirections - mean_varintrial)./mean_varintrial)*100;

removed = percentagediff < threshold_percentage;

removed = find(removed==1);
end

function trial_Processed = bintoRates(raw_Data,bin_Size,trainingFlag)
% Takes a struct of the input data, changes the bin length of each
% trial and converts to frequency. Outputs re-binned data that is zscored 
% as a struct in trial_Processed
% if training,  training_flag = 1. if not training_flag is 0.

% Movement starts 300 and ends 560

trial_Processed = struct; 
trial_Processed.bin_Size = bin_Size;
% As specified in the task
trial_Processed.start_Time = floor(320/bin_Size);
trial_Processed.end_Time = floor(560/bin_Size);
%noBins = (endTime-startTime)/bin_Size;
noTrials = size(raw_Data,1);
noDirections = size(raw_Data,2);
noNeurons = size(raw_Data(1,1).spikes,1);


for i=1:noTrials
    
    for j = 1:noDirections
        
        timePeriod = size(raw_Data(i,j).spikes,2);
        numelements = floor(timePeriod/bin_Size);
            
            for k = 1:numelements
                % Convert to hz
                    
                rate(:,k) = sum(raw_Data(i,j).spikes(:,(k-1)*(bin_Size)+1:(k*bin_Size)),2); 
                size(raw_Data(i,j).spikes,1);
               
            end
            %trial_Processed.rates(i,j).rate(l,:) = zscore(trial_Processed.rates(i,j).rate(l,:));
    
%         trial_Processed.rates(i,j).rates = zscore(rate,0,2);
        trial_Processed.rates(i,j).rates = rate;
        
        if trainingFlag == 1
            trial_Processed.rates(i,j).handPos = raw_Data(i,j).handPos;
        end
    end
end
end

function rates = gaussianKernel(data, bin_Size, window_Size, trainingFlag)
% Smooths the summed data and returns rates
% window_Size is specified in the original time step
% make standard deviation the size of 1 rate unit
% if training,  training_flag = 1. if not training_flag is 0.

% Set std to be the window_size in time scaled by the bin size and extend 
% the axes by a factor of 10.


std = window_Size/bin_Size;
range = 10*(window_Size/bin_Size);
axes = -(range-1)/2:(range-1)/2;
gauss = exp(-axes.^2/(2*(std^2)));
gauss = (1/sum(gauss))*gauss;
numTrials = size(data.rates, 1);
numDirections = size(data.rates, 2);
rates = struct; 
rates.bin_Size = bin_Size;
% As specified in the task
rates.start_Time = 320/bin_Size;
rates.end_Time = 560/bin_Size;
numNeurons = size(data.rates(1,1).rates,1);


for i=1:numTrials
    for j=1:numDirections
        numTimepts = size(data.rates(i,j).rates,2);
        rate = zeros(numNeurons, numTimepts);
       for k=1:numNeurons
        rate(k,:) = conv(data.rates(i,j).rates(k,:),gauss,'same')*(1000/bin_Size);
        
       end
       %rates.firrates(i,j).firingrates=zscore(rate,0,2);
       rates.rates(i,j).rates=rate;
       if trainingFlag == 1
        rates.rates(i,j).handPos = data.rates(i,j).handPos;
       end
    end
end   
end








    function neuron_high_MI = Entropy_Rusne(test_data, bin_size, prc)
    
    % trim and bin data
   data = trimAndBinToSum(test_data,300, 500, bin_size, 0);
    
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






    function neuron_high_MI = Entropy_Mati(test_data, bin_size, neuron_num)

noTrials = size(test_data,1);
noDirections = size(test_data,2);
noNeurons = size(test_data(1,1).spikes,1);

firing_rate_neurons = zeros(noNeurons, noDirections, noTrials);
% bin_size = 10;
noBins = 300/bin_size;

for ang = 1:noDirections
    for t = 1:noTrials
        for neuron = 1:noNeurons
            spike_train_binned = zeros(noBins,1);
            spike_train = test_data(t,ang).spikes(neuron,200:499)';

            for bin = 1:noBins
                spike_train_binned(bin) = sum(spike_train(1+bin_size*(bin-1):bin_size*bin));
            end
            spike_train_binned = gaussian_Rusne(spike_train_binned);
            firing_rate_neurons(neuron, ang, t) = mean(spike_train_binned);
        end
    end
    firing_rate_stims = mean(firing_rate_neurons, 3);
end
firing_rate = mean(firing_rate_stims, 2);

p_stim = 1/noDirections;

Mi = zeros(noNeurons, 1);

for px = 1:noNeurons
    for py = 1:noDirections
        Mi(px) = Mi(px) + firing_rate_stims(px,py) * log(firing_rate_stims(px,py) / (firing_rate(px) * p_stim));
    end 
end


Mi_index = [[1:98]', Mi];

Mi_index(any(isnan(Mi_index), 2), :) = [];
Mi_sort = sortrows(Mi_index, 2, 'descend');

th = neuron_num;

Mi_th = sortrows(Mi_sort(1:th,:), 1);

neuron_high_MI = Mi_th(:,1);

    end










  



    function smoothed = average_window(binned_data, windowSize)

% Define the size of the averaging window (adjust as needed)
% windowSize = 3; %TODO check windonw sizes
noBins = size(binned_data, 1);
% Preallocate smoothed data matrix
smoothedData = zeros(size(binned_data)); 

% Apply average window smoothing
for j = 1:noBins % Iterate over each time bin

    % Determine window boundaries
    windowStart = max(1, j - floor(windowSize/2));
    windowEnd = min(noBins, j + floor(windowSize/2));

    % Calculate the average within the window
    smoothedData(j) = mean(binned_data(windowStart:windowEnd));
end
smoothed = smoothedData;


    end


    function smoothedData = gaussian_Rusne(binnedData)

% Standard deviation of the Gaussian kernel (adjust as needed)
sigma = 2; %TODO: test different sigma

% Create the Gaussian kernel
kernelSize = 5 * sigma; % Choose an appropriate kernel size
kernel = fspecial('gaussian', [1, kernelSize], sigma);

% Preallocate smoothed data matrix
smoothedData = zeros(size(binnedData));

% Apply Gaussian smoothing to each neuron's spike train data
smoothedData(:) = conv(binnedData(:), kernel, 'same');

    end








    function smoothedData = gaussian_Rusne_whole_data_set(binnedData)


% Standard deviation of the Gaussian kernel (adjust as needed)
sigma = 2;

% Create the Gaussian kernel
kernelSize = 5 * sigma; % Choose an appropriate kernel size
kernel = fspecial('gaussian', [1, kernelSize], sigma);

% Preallocate smoothed data matrix
smoothedData = zeros(size(binnedData));

% Apply Gaussian smoothing to each neuron's spike train data
for i = 1:size(binnedData, 1) % Iterate over each neuron
    smoothedData(i, :) = conv(binnedData(i, :), kernel, 'same');
end

    end





 function trial_Processed = trimAndBinToSum(raw_Data, start_time, end_time, bin_Size, trainingFlag)
% Takes a struct of the input data
% Trims it to length that is the most relative
% Bins the spikes of each trial by summing the number of spikes in a bin_Size interval
% The dimensions are included in the struct
% if training,  training_flag = 1. if not training_flag is 0.

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
        
        if trainingFlag == 1
            trial_Processed.data(i,j).handPos = raw_Data(i,j).handPos;
        end
    end

 trial_Processed.maxBinSum = max(max(trial_Processed.trial_max));
end
    end
end
