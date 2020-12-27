% Andrea Agostinelli, Annalaura Lerede, Valentina Pacifico, Marta Sarrico

function [x, y, new_parameters] = positionEstimator(test_data, modelParameters)
% Upload Data with old values every loop

% Do check if it's the first 
len_sig = length(test_data.spikes(1,:));
new_parameters{1} = modelParameters{1};
num_angles = 8;
num_neurons = 98;
trial = new_parameters{1};
num_trials = length(trial(:,1));

% Past Positions on .decoded

if len_sig==320
    mean_test = mean(test_data.spikes');
    % Classification of angle, preset of firing rates if it's the first one
%  Best Hyperparameters: k=50, coeff=1
    coeff=1;
    k=52;
    mean_320 = zeros(num_angles,num_neurons,num_trials);
    for ang_test = 1:num_angles
        for te = 1:num_trials
            mean_320(ang_test,:,te) = mean(trial(te,ang_test).spikes(:,1:320)')';
        end
    end
    distances = zeros(num_angles,num_trials);
    for ang = 1:num_angles
        for tr = 1:num_trials
            distances(ang,tr) = power(sum(abs(power((mean_test-mean_320(ang,:,tr)),coeff))),1/coeff);
        end
    end
    [num_class, num_trials] = size(distances);
    [~, I] = sort(reshape(distances',[num_class * num_trials, 1]));
    num_nn = zeros(num_class,1);
    for i=1:k
        num_nn(ceil(I(i)/num_trials)) = num_nn(ceil(I(i)/num_trials)) + 1;
    end
    [~, new_parameters{2}] = max(num_nn);
else
    new_parameters{2} = modelParameters{2};
end

    
window = 20;
delay=0;
coeff_weight = 5;
k = 33;
coeff = 2;
eps = 1e-35;
mean_test = mean(test_data.spikes(:,len_sig-window-delay:len_sig-delay)');
angle=new_parameters{2};

% Create Firing rate of specific Window and Delay
mean_window = zeros(num_trials,num_neurons);
mask = ones(num_trials,1);
for i=1:num_trials
    if length(trial(i,angle).spikes(1,:))>=len_sig
        mean_window(i,:) = mean(trial(i,angle).spikes(:,len_sig-window-delay:len_sig-delay)');
    else
        mask(i)=0;
    end
end
k=min(k,sum(mask));
overlen=false;
if k==0
    k=5;
    overlen = true;
    mask = ones(num_trials,1);
    for i=1:num_trials
        mean_window(i,:) = mean(trial(i,angle).spikes(:,end-window-delay:end-delay)');
    end
end
% Measure k-NN
distances = ones(num_trials,1).*1e15;
for i=1:num_trials
    if mask(i)
        distances(i) = power(sum(abs(power((mean_test-mean_window(i,:)),coeff))),1/coeff);
    end
end
% Calculate Weights for Positions
[V, I] = sort(distances);
weights = zeros(k,1);
for i=1:k
    weights(i) = 1/(V(i)^coeff_weight + eps);
end
weights = weights./sum(weights);
% Calculate weighted Position
x = 0;
y = 0;
if overlen
    for i=1:k
        x = x + weights(i)*trial(I(i),angle).handPos(1,end);
        y = y + weights(i)*trial(I(i),angle).handPos(2,end);
    end
else
     for i=1:k
        x = x + weights(i)*trial(I(i),angle).handPos(1,len_sig);
        y = y + weights(i)*trial(I(i),angle).handPos(2,len_sig);
     end
end
end









