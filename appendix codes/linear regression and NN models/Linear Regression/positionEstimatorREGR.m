function [x, y] = positionEstimatorREGR(test_data, modelParameters, n, k)
XTest = test_data.spikes;
YTest_start = test_data.startHandPos;
count = 1;
num_neurons = 98;
spike_rate = zeros(num_neurons,ceil(length(XTest(1,:))/k));
for z = 1:floor(length(XTest(1,:))/k)
        spike_rate(:,z) = mean(XTest(:,count:(count+k-1)),2); %average of the rows
        count = count + k;
end
if mod(length(ceil(XTest(1,:))),k)>0
    z = z + 1;
    spike_rate(:,z) = mean(XTest(:,count:end),2);
end
pred_vel_x = modelParameters{n}.w_vx'*[ones(size(spike_rate(1,:))); spike_rate];
pred_vel_y = modelParameters{n}.w_vy'*[ones(size(spike_rate(1,:))); spike_rate];
pred_acc_x = modelParameters{n}.w_ax'*[ones(size(spike_rate(1,:))); spike_rate];
pred_acc_y = modelParameters{n}.w_ay'*[ones(size(spike_rate(1,:))); spike_rate];
pred_pos_x = modelParameters{n}.w_x'*[ones(size(spike_rate(1,:))); spike_rate];
pred_pos_y = modelParameters{n}.w_y'*[ones(size(spike_rate(1,:))); spike_rate];
pred_ax = zeros(length(pred_acc_x)+1,1);
pred_ay = zeros(length(pred_acc_y)+1,1);
pred_ax(1) = YTest_start(1,3);
pred_ay(1) = YTest_start(2,3);
pred_vx = zeros(length(pred_vel_x)+1,1);
pred_vy = zeros(length(pred_vel_y)+1,1);
pred_vx(1) = YTest_start(1,2);
pred_vy(1) = YTest_start(2,2);
pred_x = [YTest_start(1,1) pred_pos_x];
pred_y = [YTest_start(1,1) pred_pos_y];
calc = 1;
for g = 2:length(pred_vel_x)+1 % check if you can avoid evaluating all points
    pred_ax(g) = pred_x(g-1)+pred_vel_x(g-1)*k+1/2*pred_acc_x(g-1)*k^2;
    pred_ay(g) = pred_y(g-1)+pred_vel_y(g-1)*k+1/2*pred_acc_y(g-1)*k^2;
    pred_vx(g) = pred_x(g-1)+pred_vel_x(g-1)*k;
    pred_vy(g) = pred_y(g-1)+pred_vel_y(g-1)*k;
    calc = calc + k;
end
x = [pred_x(end) pred_vx(end) pred_ax(end)];
y = [pred_y(end) pred_vy(end) pred_ay(end)];
end