clear
clc

load("monkeydata_training.mat");
%% Firing rates of neurons (equivalent to probabilities of firing)

firing_rate_neurons = zeros(98, 8, 100);
bin_size = 10;


for ang = 1:8
    for t = 1:100
        for neuron = 1:98
            spike_train_binned = zeros(60,1);
            spike_train = trial(t,ang).spikes(neuron,200:499)';
            for bin = 1:(300/bin_size)
                spike_train_binned(bin) = sum(spike_train(1+bin_size*(bin-1):bin_size*bin));
            end
            firing_rate_neurons(neuron, ang, t) = mean(spike_train_binned);
        end
    end
    firing_rate_stims = mean(firing_rate_neurons, 3);
end
firing_rate = mean(firing_rate_stims, 2);

%%
p_stim = 1/8;

Mi = zeros(98, 1);

for px = 1:98
    for py = 1:8
        Mi(px) = Mi(px) + firing_rate_stims(px,py) * log(firing_rate_stims(px,py) / (firing_rate(px) * p_stim));
    end 
end


Mi_index = [[1:98]', Mi];

Mi_index(any(isnan(Mi_index), 2), :) = [];
Mi_sort = sortrows(Mi_index, 2, 'descend');

th = 40;

Mi_th = sortrows(Mi_sort(1:th,:), 1);






