clc
clear all

% Dataset path
load('/Users/agos/Downloads/monkeydata_training.mat')

num_trials = 100;
num_angles = 8;
num_neurons = 98;
coeff = [1];
final_acc = zeros(num_angles,1);
num_k = 52;
num_test = 1;
acc=zeros(length(coeff),length(num_k),num_test);
final_acc=zeros(length(coeff),length(num_k),num_test,num_angles);
conf=zeros(2,30,8);
rng(201323);

for test = 1:num_test
    index = randperm(num_trials,num_trials);
    index_train = index(1:floor(num_trials*0.7));
    index_test = index(floor(num_trials*0.7) + 1:end);

    trainBOX = zeros(num_angles,num_neurons,length(index_train));
    testBOX = zeros(num_angles,num_neurons,length(index_test));
    tic
    for n = 1:num_neurons
        for ang_test = 1:num_angles
            for te = 1:length(index_test)
                testBOX(ang_test,n,te) = mean(trial(index_test(te),ang_test).spikes(n,1:320));
            end
        end
    end

    for n = 1:num_neurons
        for ang_train = 1:num_angles
            for tr = 1:length(index_train)
                trainBOX(ang_train,n,tr) = mean(trial(index_train(tr),ang_train).spikes(n,1:320));
            end
        end
    end

    % Calculate Accuracy

    for i=1:length(coeff)
        for j=1:length(num_k)
            for ang_test = 1:num_angles
                for te= 1:length(index_test)
                    distance = zeros(num_angles,length(index_train));
                    for ang_train = 1:num_angles
                        for tr = 1:length(index_train)
                            distance(ang_train,tr) = distance(ang_train,tr) + get1Dist_ISIH(testBOX(ang_test,:,te), trainBOX(ang_train,:,tr), coeff(i));
                        end
                    end
                    conf(1,te,ang_test) = classify(distance, num_k(j));
                    conf(2,te,ang_test) = ang_test;
                    if conf(1,te,ang_test)==ang_test
                        final_acc(i,j,test,ang_test) = final_acc(i,j,test,ang_test)+1;
                    end
                end
            end
            final_acc(i,j,test,:) = final_acc(i,j,test,:)./length(index_test);
            acc(i,j,test) = mean(final_acc(i,j,test,:));
        end
    end
    toc
end

