function neuron_high_MI = Entropy(test_data)
    
    % trim and bin data
    data = trimAndBinToSum(test_data,200, 500, 10, 0);
    
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
    median_MI = median (MI);
    neuron_high_MI = find (MI > median_MI);

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