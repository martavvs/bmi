clc
clear all

% Dataset Path
load '/Users/agos/Downloads/monkeydata_training.mat'

% Looking for Hyperparameters

% Exponent to calculate distances between neighbours in the classification
% model
coeffClass = [1];
lCC = length(coeffClass);

% Number k of neighbours in the classification
kClass = [52];
lkC = length(kClass);

% Exponent to calculate distances between neighbours in the regression
% model
coeffReg = [2];
lcR = length(coeffReg);

% Number k of neighbours in the regression
kReg = [33];
lkR = length(kReg);

% Coefficient to calculate the weighted averaged position among k neighbours
coeffWeights = [5];
lcW = length(coeffWeights);

% Length of past signal to feed the 'Regression' Model
window = 20;

% Select training, testing and validation data 
rng(2019);

ix = randperm(length(trial));
trainingData = trial(ix(1:65),:);
validationData = trial(ix(66:80),:);
testData = trial(ix(81:end),:);
    

% Parameters Storage
errors = zeros(lCC,lkC,lcR,lkR,lcW);
timess = zeros(lCC,lkC,lcR,lkR,lcW);

% Best Parameters found to use for TestSet
bestPar = zeros(5,1);
bestErr = 1e10;

for index_coeff_Class = 1:lCC
    for index_k_Class = 1:lkC
        for index_coeff_Regr = 1:lcR
            for index_k_Regr = 1:lkR
                for index_coeff_Weights = 1:lcW
                    % Set random number generator
                    tic

                    fprintf('Testing the continuous position estimator...')

                    meanSqError = 0;
                    n_predictions = 0;  
                    
                    % Train Model
                    modelParameters = positionEstimatorTraining(trainingData);

                    for tr=1:size(validationData,1)
                        display(['Decoding block ',num2str(tr),' out of ',num2str(size(validationData,1))]);
                        pause(0.001)
                        for direc=randperm(8) 
                            decodedHandPos = [];
                            times=320:20:size(validationData(tr,direc).spikes,2);

                            for t=times
                                past_current_trial.trialId = validationData(tr,direc).trialId;
                                past_current_trial.spikes = validationData(tr,direc).spikes(:,1:t); 
                                past_current_trial.decodedHandPos = decodedHandPos;

                                past_current_trial.startHandPos = validationData(tr,direc).handPos(1:2,1); 
                                test_data = past_current_trial;
                                len_sig = length(test_data.spikes(1,:));
                                new_parameters{1} = modelParameters{1};
                                num_angles = 8;
                                num_neurons = 98;
                                ttrial = new_parameters{1};
                                num_trials = length(ttrial(:,1));

                                % Past Positions on .decoded

                                if len_sig==320
                                    mean_test = mean(test_data.spikes');
                                    % Classification of angle, preset of firing rates if it's the first one
                                    coeff=coeffClass(index_coeff_Class);
                                    k=kClass(index_k_Class);
                                    mean_320 = zeros(num_angles,num_neurons,num_trials);
                                    for ang_test = 1:num_angles
                                        for te = 1:num_trials
                                            mean_320(ang_test,:,te) = mean(ttrial(te,ang_test).spikes(:,1:320)')';
                                        end
                                    end
                                    distances = zeros(num_angles,num_trials);
                                    for ang = 1:num_angles
                                        for tra = 1:num_trials
                                            distances(ang,tra) = power(sum(abs(power((mean_test-mean_320(ang,:,tra)),coeff))),1/coeff);
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

                                delay=0;
                                coeff_weight = coeffWeights(index_coeff_Weights);
                                k = kReg(index_k_Regr);
                                coeff = coeffReg(index_coeff_Regr);
                                eps = 1e-35;
                                mean_test = mean(test_data.spikes(:,len_sig-window-delay:len_sig-delay)');
                                angle=new_parameters{2};

                                % Create Firing rate of specific Window and Delay
                                mean_window = zeros(num_trials,num_neurons);
                                mask = ones(num_trials,1);
                                for i=1:num_trials
                                    if length(ttrial(i,angle).spikes(1,:))>=len_sig
                                        mean_window(i,:) = mean(ttrial(i,angle).spikes(:,len_sig-window-delay:len_sig-delay)');
                                    else
                                        mask(i)=0;
                                    end
                                end
                                k=min(k,sum(mask));
                                overlen=false;
                                if k==0
                                    k=kReg(index_k_Regr);
                                    overlen = true;
                                    mask = ones(num_trials,1);
                                    for i=1:num_trials
                                        mean_window(i,:) = mean(ttrial(i,angle).spikes(:,end-window-delay:end-delay)');
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
                                decodedPosX = 0;
                                decodedPosY = 0;
                                if overlen
                                    for i=1:k
                                        decodedPosX = decodedPosX + weights(i)*ttrial(I(i),angle).handPos(1,end);
                                        decodedPosY = decodedPosY + weights(i)*ttrial(I(i),angle).handPos(2,end);
                                    end
                                else
                                     for i=1:k
                                        decodedPosX = decodedPosX + weights(i)*ttrial(I(i),angle).handPos(1,len_sig);
                                        decodedPosY = decodedPosY + weights(i)*ttrial(I(i),angle).handPos(2,len_sig);
                                     end
                                end
                                modelParameters = new_parameters;
                                decodedPos = [decodedPosX; decodedPosY];
                                decodedHandPos = [decodedHandPos decodedPos];
                                meanSqError = meanSqError + norm(validationData(tr,direc).handPos(1:2,t) - decodedPos)^2;

                            end
                            n_predictions = n_predictions+length(times);
                            hold on
                        end
                    end
                    sqrt(meanSqError/n_predictions)
                    errors(index_coeff_Class,index_k_Class,index_coeff_Regr,index_k_Regr,index_coeff_Weights) = sqrt(meanSqError/n_predictions);
                    if sqrt(meanSqError/n_predictions) < bestErr
                        bestErr = sqrt(meanSqError/n_predictions);
                        bestPar = [coeffClass(index_coeff_Class),kClass(index_k_Class),coeffReg(index_coeff_Regr),kReg(index_k_Regr),coeffWeights(index_coeff_Weights)];
                    rmpath(genpath('ff'))
                    toc
                    timess(index_coeff_Class,index_k_Class,index_coeff_Regr,index_k_Regr,index_coeff_Weights) = toc;
                    end
                end
            end
        end
    end
end

% Set random number generator
tic

fprintf('Evaluating the Test Data with The Best Hyperparameters\n')

meanSqError = 0;
n_predictions = 0;  

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];
        times=320:20:size(testData(tr,direc).spikes,2);

        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            test_data = past_current_trial;
            len_sig = length(test_data.spikes(1,:));
            new_parameters{1} = modelParameters{1};
            num_angles = 8;
            num_neurons = 98;
            ttrial = new_parameters{1};
            num_trials = length(ttrial(:,1));

            % Past Positions on .decoded

            if len_sig==320
                mean_test = mean(test_data.spikes');
                % Classification of angle, preset of firing rates if it's the first one
                coeff=bestPar(1);
                k=bestPar(2);
                mean_320 = zeros(num_angles,num_neurons,num_trials);
                for ang_test = 1:num_angles
                    for te = 1:num_trials
                        mean_320(ang_test,:,te) = mean(ttrial(te,ang_test).spikes(:,1:320)')';
                    end
                end
                distances = zeros(num_angles,num_trials);
                for ang = 1:num_angles
                    for tra = 1:num_trials
                        distances(ang,tra) = power(sum(abs(power((mean_test-mean_320(ang,:,tra)),coeff))),1/coeff);
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

            delay=0;
            coeff_weight = bestPar(5);
            k=bestPar(4);
            coeff = bestPar(3);
            eps = 1e-35;
            mean_test = mean(test_data.spikes(:,len_sig-window-delay:len_sig-delay)');
            angle=new_parameters{2};

            % Create Firing rate of specific Window and Delay
            mean_window = zeros(num_trials,num_neurons);
            mask = ones(num_trials,1);
            for i=1:num_trials
                if length(ttrial(i,angle).spikes(1,:))>=len_sig
                    mean_window(i,:) = mean(ttrial(i,angle).spikes(:,len_sig-window-delay:len_sig-delay)');
                else
                    mask(i)=0;
                end
            end
            k=min(k,sum(mask));
            overlen=false;
            if k==0
                k=kReg(index_k_Regr);
                overlen = true;
                mask = ones(num_trials,1);
                for i=1:num_trials
                    mean_window(i,:) = mean(ttrial(i,angle).spikes(:,end-window-delay:end-delay)');
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
            decodedPosX = 0;
            decodedPosY = 0;
            if overlen
                for i=1:k
                    decodedPosX = decodedPosX + weights(i)*ttrial(I(i),angle).handPos(1,end);
                    decodedPosY = decodedPosY + weights(i)*ttrial(I(i),angle).handPos(2,end);
                end
            else
                 for i=1:k
                    decodedPosX = decodedPosX + weights(i)*ttrial(I(i),angle).handPos(1,len_sig);
                    decodedPosY = decodedPosY + weights(i)*ttrial(I(i),angle).handPos(2,len_sig);
                 end
            end
            modelParameters = new_parameters;
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

        end
        n_predictions = n_predictions+length(times);
        hold on
    end
end
final_error = sqrt(meanSqError/n_predictions)
toc