function modelParameters = positionEstimatorTrainingREG(training_data, k, delay)
num_neurons = length(training_data(1,1).spikes(:,1));
modelParameters = cell(8,1);
for n = 1:8
    num_trials = length(training_data(:,n));
    for h = 1:num_trials % to shift all hand position to a delay to consider the time the signals take to travel from the brain to the arm
        for j = 1:length(training_data(h,n).handPos(1,1:end-delay))
            training_data(h,n).handPos(:,j) = training_data(h,n).handPos(:,j+delay);
        end
    end
    for h = 1:num_trials
        training_data(h,n).handPos = training_data(h,n).handPos(:,1:end-delay);
        training_data(h,n).spikes = training_data(h,n).spikes(:,1:end-delay);
    end
    sum = 0;
    for q = 1:num_trials
        sum = sum + ceil(length(training_data(q,n).spikes(1,:))/k);
    end
    mean_k_spikes = zeros(num_neurons,sum);
    pos_y = zeros(2,sum);
    v_y = zeros(2,sum);
    a_y = zeros(2,sum);
    t = 1; % count external to the trials
    for i = 1:num_trials
        count = 1; % count internal to the trial
        for z = 1:floor(length(training_data(i,n).spikes(1,:))/k)
            mean_k_spikes(:,t) = mean(training_data(i,n).spikes(:,count:(count+k-1)),2); % spike rate for k time steps
            pos_y(:,t) = mean(training_data(i,n).handPos(1:2,count:(count+k-1)),2);
            v_y(:,t) = (training_data(i,n).handPos(1:2,count+k-1)-training_data(i,n).handPos(1:2,count))/k; % velocity for k time steps
            if mod(k,2)>0
                a_y(:,t) = ((training_data(i,n).handPos(1:2,count+k-1)-training_data(i,n).handPos(1:2,count+(k+1)/2-1))/((k+1)/2)-(training_data(i,n).handPos(1:2,count+(k+1)/2-1)-training_data(i,n).handPos(1:2,count))/((k+1)/2))/k;
            else
                a_y(:,t) = (((training_data(i,n).handPos(1:2,count+k-1)-training_data(i,n).handPos(1:2,count+k/2))/k/2)-(training_data(i,n).handPos(1:2,count+k/2-1)-training_data(i,n).handPos(1:2,count))/k/2)/k;
            end
            count = count + k;
            t = t + 1;
        end
        if mod(length(training_data(i,n).spikes(1,:)),k)>0 % if at the end of the trial you have less than k elements
            mean_k_spikes(:,t) = mean(training_data(i,n).spikes(:,count:end),2);
            pos_y(:,t) = mean(training_data(i,n).handPos(1:2,count:end),2);
            v_y(:,t) = (training_data(i,n).handPos(1:2,end)-training_data(i,n).handPos(1:2,count))/mod(length(training_data(i,n).spikes(1,:)),k);
            if mod(mod(length(training_data(i,n).spikes(1,:)),k),2)>0
                a_y(:,t) = ((training_data(i,n).handPos(1:2,end)-training_data(i,n).handPos(1:2,end-(end-count)/2))/(mod(length(training_data(i,n).spikes(1,:)),k)+1)/2-(training_data(i,n).handPos(1:2,end-(end-count)/2)-training_data(i,n).handPos(1:2,count))/(mod(length(training_data(i,n).spikes(1,:)),k)+1)/2)/k;
            else
                a_y(:,t) = ((training_data(i,n).handPos(1:2,end)-training_data(i,n).handPos(1:2,end-(end-count+1)/2+1))/(mod(length(training_data(i,n).spikes(1,:)),k))/2-(training_data(i,n).handPos(1:2,end-(end-count+1)/2)-training_data(i,n).handPos(1:2,count))/(mod(length(training_data(i,n).spikes(1,:)),k))/2)/k;
            end
            t = t + 1;
        end
    end
    mean_k_spikes = mean_k_spikes';
    mean_k_spikes = [ones(size(mean_k_spikes(:,1))),mean_k_spikes]; % add bias ones
    labels_x = pos_y(1,:)';
    labels_y = pos_y(2,:)';
    labels_vx = v_y(1,:)';
    labels_vy = v_y(2,:)';
    labels_ax = a_y(1,:)';
    labels_ay = a_y(2,:)';
    w_x = regress(labels_x,mean_k_spikes); % linear regression for vel x
    w_y = regress(labels_y,mean_k_spikes); % for y
    w_vx = regress(labels_vx,mean_k_spikes); % linear regression for vel x
    w_vy = regress(labels_vy,mean_k_spikes); % for y
    w_ax = regress(labels_ax,mean_k_spikes); % linear regression for acc x
    w_ay = regress(labels_ay,mean_k_spikes); % for y
    modelParameters{n,1}.w_x = w_x;
    modelParameters{n,1}.w_y = w_y;
    modelParameters{n,1}.w_ax = w_ax;
    modelParameters{n,1}.w_ay = w_ay;
    modelParameters{n,1}.w_vx = w_vx;
    modelParameters{n,1}.w_vy = w_vy;
end
end