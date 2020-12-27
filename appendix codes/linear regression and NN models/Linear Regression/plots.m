% RMSE TimeSteps for different angles
figure
ave_mse = mean(RMS_FIN,4);
for n = 1:8
    subplot(4,4,n)
    plot(RMS_FIN(2,:,2,n),'r')
    hold on
    plot(RMS_FIN(2,:,1,n),'g')
    hold on
    plot(RMS_FIN(2,:,3,n),'b')
end
% RMSE TimeDelays for different angles
figure
for n = 1:8
    subplot(4,4,n)
    plot(RMS_FIN(:,20,2,n),'r')
    hold on
    plot(RMS_FIN(:,20,1,n),'g')
    hold on
    plot(RMS_FIN(:,20,3,n),'b')
end
err_delay1 = zeros(40,1);
err_delay2 = zeros(40,1);
err_delay3 = zeros(40,1);
for i = 1:40
    err_delay2(i) = max(RMS_FIN(i,20,2,:))-min(RMS_FIN(i,20,2,:));
    err_delay1(i) = max(RMS_FIN(i,20,1,:))-min(RMS_FIN(i,20,1,:));
    err_delay3(i) = max(RMS_FIN(i,20,3,:))-min(RMS_FIN(i,20,3,:));
end
% RMSE TimeDelays for average
figure
subplot(1,2,1)
errorbar(1:40,ave_mse(:,20,1),err_delay1,'-b','LineWidth',1,'DisplayName','Position')
hold on
errorbar(1:40,ave_mse(:,20,3),err_delay3,'-og','LineWidth',1.2,'DisplayName','Acceleration')
hold on
errorbar(1:40,ave_mse(:,20,2),err_delay2,'-r','LineWidth',1,'DisplayName','Velocity')
err_k1 = zeros(20,1);
err_k2 = zeros(20,1);
err_k3 = zeros(20,1);
xlabel('Time Delays')
ylabel('Model RMSE')
title('Choice of Hyperparameter: Time Delays')
legend('Location','northwest')
for i = 1:20
    err_k2(i) = max(RMS_FIN(2,i,2,:))-min(RMS_FIN(2,i,2,:));
    err_k1(i) = max(RMS_FIN(2,i,1,:))-min(RMS_FIN(2,i,1,:));
    err_k3(i) = max(RMS_FIN(2,i,3,:))-min(RMS_FIN(2,i,3,:));
end
% RMSE TimeSteps for average
figure
subplot(1,2,2)
errorbar(1:20,ave_mse(2,:,1),err_k1,'-b','LineWidth',1,'DisplayName','Position')
hold on
errorbar(1:20,ave_mse(2,:,3),err_k3,'-og','LineWidth',1.2,'DisplayName','Acceleration')
hold on
errorbar(1:20,ave_mse(2,:,2),err_k2,'-r','LineWidth',1,'DisplayName','Velocity')
xlabel('Time Steps')
ylabel('Model RMSE')
title('Choice of Hyperparameter: Time Steps')
legend('Location','southwest')
% color map Delays combo Steps
figure
imagesc(ave_mse(:,:,2))
title('Velocity - Linear Regression Model RMSE')
xlabel('Time Steps')
ylabel('Time Delays')
colorbar()
% surface map Delays combo Steps
figure
surf(ave_mse(:,:,2))
colormap default
title('Velocity - Linear Regression Model')
xlabel('Time Steps')
ylabel('Time Delays')
zlabel('RMSE')
colorbar()