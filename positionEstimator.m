% This code is submitted my BodyMassIndex:
% Jamie Shing Him Ho
% Mateusz Chodkowski
% Rusne Joneikyte
% Cassius Kua

function [x, y] = positionEstimator(test_data, modelParameters)
%% KNN
noNeurons_filtered = modelParameters.noNeurons_filtered;
neurons_to_keep = modelParameters.neurons_to_keep;

mean_firing_rates = zeros(1, noNeurons_filtered);
i=0;

% Calculate mean firing rates
for neuron = 1:modelParameters.noNeurons
    if ismember(neuron, neurons_to_keep)
        i = i + 1;        
        mean_firing_rates(i) = sqrt(mean(test_data.spikes(neuron,:)));
    end
end

%% Regression
% Predict angle using KNN
test_direction = KNN(modelParameters.knn_firing_rates', modelParameters.knn_directions, mean_firing_rates, modelParameters.optimalK);

% Calculate 't' based on test_data.spikes
time = size(test_data.spikes, 2);
t = min(floor(time / 20) -15, 26);

% PCA and LDA transformations
pca_coeff = modelParameters.PCA(test_direction, t).coeff;
lda_coeff = modelParameters.LDA(test_direction, t).coeff;
pca_transformed = mean_firing_rates * pca_coeff;
lda_transformed = pca_transformed * lda_coeff;

% Predict x and y
x = [1, lda_transformed] * modelParameters.regression(test_direction,t).coef_x;
y = [1, lda_transformed] * modelParameters.regression(test_direction,t).coef_y;
end

%% FUNCTIONS

function prediction = KNN(xtrain, ytrain, xtest, k)

prediction = zeros(size(xtest, 1), 1);

for xxtest = 1:size(xtest, 1)
    distances = sqrt(sum((xtrain - xtest(xxtest, :)).^2, 2));
    [~, indices_sorted] = sort(distances, 'ascend');
    top_k_indices = indices_sorted(1:k);
    prediction(xxtest) = mode(ytrain(top_k_indices));
end

end


