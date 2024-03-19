function [modelParameters] = positionEstimatorTraining(training_data)
n_conditions = 8;
final_train_data = struct;
modelParameters = struct;
train_data_len = length(training_data);
knn_data = struct;

%% Preprocessing
% Remove low variance neurons
[to_remove, percentagediff]=removeVariance(training_data,5);
modelParameters.to_remove = to_remove;


% organise train data into struct based on incresing window size 
count = 1;
c = 1;
for cond = 1:n_conditions
    for i = 1:train_data_len
        
         times = 320:20:560;
       
        for l = times 
            mean_activity = [];
            for neuron = 1:98
                if ~ismember(neuron, to_remove)
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
neurons_len = 98 - length(to_remove);
modelParameters.neurons_len = neurons_len;
data_len = length(final_train_data);
angles = zeros(data_len, 1);
elapsed_times = zeros(data_len, 1);
lambda = 0.001;
pcaDimensions = 10; % Number of dimensions to keep after PCA


% Prepare data for PCA and LDA
for i = 1:data_len
    angles(i) = final_train_data(i).angle;
    elapsed_times(i) = final_train_data(i).elapsed_time;
end

% Apply PCA and LDA per angle and time
for a = 1:8
    for t = 1:13
        my_t = t*20 + 300;
        idx = (angles == a) & (elapsed_times == my_t);
        pca_matrix = [final_train_data(idx).mean_activity];
        pca_matrix = reshape(pca_matrix, [], neurons_len);

        % Applying PCA
        [coeff, score] = my_pca(pca_matrix, pcaDimensions);
        anglesVector = [final_train_data(idx).angle];

        % Applying LDA with regularization
        ldaScores = mylda_simple(score, anglesVector);

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
    for t = 1:13
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
function [coefficients, scores] = my_pca(mean_firing_rates, dim)
    % Center the mean firing rate matrix
    centered_mean_firing_rates = mean_firing_rates - mean(mean_firing_rates);

    % Scale the matrix by dividing each column by its standard deviation
    scaled_mean_firing_rates = centered_mean_firing_rates ./ std(centered_mean_firing_rates);

    % Run PCA on the scaled mean firing rate matrix
    [coefficients, scores, eigenvalues] = my_pca2(scaled_mean_firing_rates,dim);
end

% Coded with reference to 
% https://www.youtube.com/watch?v=FgakZw6K1QQ
function [coefficients, scores, eigenvalues] = my_pca2(X, num_components)
    % Compute the covariance matrix
    covariance_matrix = cov(X);

    % Compute the eigenvectors and eigenvalues of the covariance matrix
    [coefficients, eigenvalues] = eig(covariance_matrix);

    % Sort the eigenvectors and eigenvalues in descending order of eigenvalue
    eigenvalues = diag(eigenvalues);
    [eigenvalues, sorted_indices] = sort(eigenvalues, 'descend');
    coefficients = coefficients(:, sorted_indices);

    % Compute the principal component scores
    scores = X * coefficients;
    scores = scores(:, 1:num_components);
    coefficients = coefficients(:, 1:num_components);
    eigenvalues = eigenvalues(1:num_components);
end


%% lda function

function ldaScores = mylda_simple(pcaScores, classLabels)
    % Ensure classLabels are column vectors
    if isrow(classLabels)
        classLabels = classLabels';
    end
    
    % Calculate the mean of the whole dataset
    overallMean = mean(pcaScores, 1);
    
    % Initialize within-class scatter matrix Sw and between-class scatter matrix Sb
    Sw = zeros(size(pcaScores, 2), size(pcaScores, 2));
    Sb = zeros(size(pcaScores, 2), size(pcaScores, 2));
    
    uniqueClasses = unique(classLabels);
    numClasses = length(uniqueClasses);
    
    % Calculate Sw and Sb
    for i = 1:numClasses
        % Indices of rows for the current class
        classIndices = classLabels == uniqueClasses(i);
        
        % Scores for the current class
        classScores = pcaScores(classIndices, :);
        
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
    
    % Transform the dataset
    ldaTransformedData = pcaScores * sortedEigenvectors;
    
    % Store results
    ldaScores.coeff = sortedEigenvectors;
    ldaScores.transformed_data = ldaTransformedData;
    ldaScores.eigenvalues = sortedEigenvalues;
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
function [removed,percentagediff] = removeVariance(training_data,threshold_percentage)
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
end
