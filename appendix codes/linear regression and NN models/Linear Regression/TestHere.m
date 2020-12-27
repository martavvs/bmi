clear all
load('/Users/lau/Downloads/monkeydata_training.mat');

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);
RMS_FIN = zeros(40,20,3,8);
for n = 1:8
    flag = 1;
    for delay = 1:5:200
        for k = 1:20
    %         fprintf('Testing the continuous position estimator...')
            meanSqError = 0;
            meanSqErrorV = 0;
            meanSqErrorA = 0;
            n_predictions = 0;  

    %         figure
    %         hold on
    %         axis square
    %         grid

            % Train Model
            modelParameters = positionEstimatorTrainingREG(trainingData, k, delay);

            for tr=1:size(testData,1)
    %             display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    %             pause(0.001)
                direc = n;
                %for direc=randperm(8) 
                    decodedHandPos = [];
                    decodedHandPosV = [];
                    decodedHandPosA = [];

                    times=320:20:size(testData(tr,direc).spikes,2);

                    for t=times
                        past_current_trial.trialId = testData(tr,direc).trialId;
                        past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                        past_current_trial.decodedHandPos = decodedHandPos;

                        if t == 320
                            past_current_trial.startHandPos(:,1) = testData(tr,direc).handPos(1:2,1);
                            past_current_trial.startHandPos(:,2) = testData(tr,direc).handPos(1:2,1);
                            past_current_trial.startHandPos(:,3) = testData(tr,direc).handPos(1:2,1);
                        else
                            past_current_trial.startHandPos(:,1) = decodedHandPos(1:2,end);
                            past_current_trial.startHandPos(:,2) = decodedHandPosV(1:2,end);
                            past_current_trial.startHandPos(:,3) = decodedHandPosA(1:2,end);
                        end
                        [decodedPosX, decodedPosY] = positionEstimatorREGR(past_current_trial,modelParameters,n,k);
                        decodedPos = [decodedPosX(1); decodedPosY(1)];
                        decodedHandPos = [decodedHandPos decodedPos];
                        decodedPosV = [decodedPosX(2); decodedPosY(2)];
                        decodedPosA = [decodedPosX(3); decodedPosY(3)];
                        decodedHandPosV = [decodedHandPosV decodedPosV];
                        decodedHandPosA = [decodedHandPosA decodedPosA];

                        meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                        meanSqErrorV = meanSqErrorV + norm(testData(tr,direc).handPos(1:2,t) - decodedPosV)^2;
                        meanSqErrorA = meanSqErrorA + norm(testData(tr,direc).handPos(1:2,t) - decodedPosA)^2;

                    end
                    n_predictions = n_predictions+length(times);
    %                 hold on
    %                  plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
    %                  plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    %             %end
            end
            RMSE1 = sqrt(meanSqError/n_predictions);
            RMSE1V = sqrt(meanSqErrorV/n_predictions);
            RMSE1A = sqrt(meanSqErrorA/n_predictions);
            RMS_FIN(flag,k,1,n) = RMSE1;
            RMS_FIN(flag,k,2,n) = RMSE1V;
            RMS_FIN(flag,k,3,n) = RMSE1A;
        end
        flag = flag + 1;
    end
end
