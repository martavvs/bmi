%% 
clc
close all
clear all
%% Load Data
load('monkeydata_training.mat');
ang_1 = trial(:,1);


%% Split into Train and Test 
tot_trial = length(trial(:,1));
p = .7      % proportion of rows to select for training
tf = false(tot_trial,1)    % create logical index vector
tf(1:round(p*tot_trial)) = true     
tf = tf(randperm(tot_trial))   % randomise order
Traindata = ang_1(tf,:) 
Testdata = ang_1(~tf,:)

Traindata=struct2cell(Traindata);
Testdata=struct2cell(Testdata);
%% Obtain XTrain and YTrain
XTrain = Traindata(2,:)';
YTrain_all = Traindata(3,:)';
nb_trial = length(Traindata);
YTrain_x={};
YTrain_y={};
for i=1:nb_trial
    YTrain_x{i} = YTrain_all{i,1}(1,:);
    YTrain_y{i} = YTrain_all{i,1}(2,:);
end
YTrain_x = YTrain_x';
YTrain_y = YTrain_y';

%% Put sets in descending sequence order by length

for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');

XTrain = XTrain(idx);
YTrain_y = YTrain_y(idx);
%% Network Architecture

numResponses = size(YTrain_y{1},1);
featureDimension = size(XTrain{1},1);
numHiddenUnits = 45;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

%% Train Model
net = trainNetwork(XTrain,YTrain_y,layers,options);

%% Obtain XTest and YTest
XTest = Testdata(2,:)';
YTest_all = Testdata(3,:)';

nb_trial2 = length(Testdata);
YTest_x={};
YTest_y={};
for i=1:nb_trial2
    YTest_x{i} = YTest_all{i,1}(1,:);
    YTest_y{i} = YTest_all{i,1}(2,:);
end

YTest_x = YTest_x';
YTest_y = YTest_y';

%% Prediction
YPred = predict(net,XTest,'MiniBatchSize',1);

%% 4 random trajectories predicted
idx = randperm(numel(YPred),4);
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(YTest_y{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([-30 70])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("X Position")
end
legend(["Test Data" "Predicted"],'Location','southeast')

%% MSE
 mse = zeros(30,1);
for i = 1:numel(YTest_x)
    mse1 = 0;
    for j = 1:numel(YTest_x{i})
        YTestvalue = YTest_x{i}(j);
        YPredvalue = YPred{i}(j);
        %rmse(j) = sqrt(mean((YPredvalue - YTestvalue).^2));
        mse1 = mse1 + norm(YTestvalue-YPredvalue)^2;
    end
    mse(i) = sqrt(mse1/numel(YTest_x{i}));
end


figure
plot(mse)
