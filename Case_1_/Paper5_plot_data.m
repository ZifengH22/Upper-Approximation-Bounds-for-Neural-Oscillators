clear;clc;
close all;
loss_L = 8;
hidden_number = 5;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'E_X_train.mat']);
load([Data_path,'E_X_pred_',num2str(hidden_number),'.mat']);

load([Data_path,'Train_epochs_loss_',num2str(hidden_number),'.mat']);
load([Data_path,'Val_epochs_loss_',num2str(hidden_number),'.mat']);
load([Data_path,'Train_epochs_loss_mse_',num2str(hidden_number),'.mat']);
load([Data_path,'Val_epochs_loss_mse_',num2str(hidden_number),'.mat']);
load([Data_path,'Train_epochs_loss_max_',num2str(hidden_number),'.mat']);
load([Data_path,'Val_epochs_loss_max_',num2str(hidden_number),'.mat']);
load([Data_path,'RK4GRUcell_derivative_norm_',num2str(hidden_number),'.mat']);
load([Data_path,'top_DNN_derivative_norm_',num2str(hidden_number),'.mat']);

eval(['E_X_pred = E_X_pred_',num2str(hidden_number),';']);
eval(['Train_epochs_loss = Train_epochs_loss_',num2str(hidden_number),';']);
eval(['Val_epochs_loss = Val_epochs_loss_',num2str(hidden_number),';']);
eval(['Train_epochs_loss_mse = Train_epochs_loss_mse_',num2str(hidden_number),';']);
eval(['Val_epochs_loss_mse = Val_epochs_loss_mse_',num2str(hidden_number),';']);
eval(['Train_epochs_loss_max = Train_epochs_loss_max_',num2str(hidden_number),';']);
eval(['Val_epochs_loss_max = Val_epochs_loss_max_',num2str(hidden_number),';']);
eval(['RK4GRUcell_derivative_norm = RK4GRUcell_derivative_norm_',num2str(hidden_number),';']);
eval(['top_DNN_derivative_norm = top_DNN_derivative_norm_',num2str(hidden_number),';']);

%%%%%%%%%%%%%%%%%Training and validation loss plots%%%%%%%%%%%%%%%%%%%%%%%
figure(str2num([num2str(hidden_number),'1']))
semilogy(Train_epochs_loss(:,1),Train_epochs_loss(:,2),'k-',Val_epochs_loss(:,1),Val_epochs_loss(:,2),'b-.');
legend('Training loss (Runge-Kutta neural oscillator)','Validation loss (Runge-Kutta neural oscillator)')
% xlim([0,5000]);
% ylim([1e-3,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);

Train_epochs_loss_mse_plot = Train_epochs_loss_mse(:,2:end)';
Train_epochs_loss_mse_plot = Train_epochs_loss_mse_plot(:);
Val_epochs_loss_mse_plot = Val_epochs_loss_mse(:,2:end)';
Val_epochs_loss_mse_plot = Val_epochs_loss_mse_plot(:);
figure(str2num([num2str(hidden_number),'2']))
semilogy([1:length(Train_epochs_loss_mse_plot)],Train_epochs_loss_mse_plot,'k-');
xlim([0,length(Train_epochs_loss_mse_plot)]);
% ylim([1e-1,1e2]);
xlabel('Number of iterations');
ylabel('Loss mse');
set(gca,'fontsize',12);
figure(str2num([num2str(hidden_number),'3']))
semilogy([1:length(Val_epochs_loss_mse_plot)],Val_epochs_loss_mse_plot,'b-.');
xlim([0,length(Val_epochs_loss_mse_plot)]);
% ylim([1e-1,1e2]);
xlabel('Epoch');
ylabel('Loss mse');
set(gca,'fontsize',12);

Train_epochs_loss_max_plot = Train_epochs_loss_max(:,2:end)';
Train_epochs_loss_max_plot = Train_epochs_loss_max_plot(:);
Val_epochs_loss_max_plot = Val_epochs_loss_max(:,2:end)';
Val_epochs_loss_max_plot = Val_epochs_loss_max_plot(:);
figure(str2num([num2str(hidden_number),'4']))
plot([1:length(Train_epochs_loss_max_plot)],Train_epochs_loss_max_plot,'k-');
xlim([0,length(Train_epochs_loss_max_plot)]);
% ylim([0,1e2]);
xlabel('Number of iterations');
ylabel('Loss max');
set(gca,'fontsize',12);
figure(str2num([num2str(hidden_number),'5']))
plot([1:length(Val_epochs_loss_max_plot)],Val_epochs_loss_max_plot,'b-.');
xlim([0,length(Val_epochs_loss_max_plot)]);
% ylim([0,1e2]);
xlabel('Epoch');
ylabel('Loss max');
set(gca,'fontsize',12);


RK4GRUcell_derivative_norm_plot = RK4GRUcell_derivative_norm(:,2:end)';
RK4GRUcell_derivative_norm_plot = RK4GRUcell_derivative_norm_plot(:);
top_DNN_derivative_norm_plot = top_DNN_derivative_norm(:,2:end)';
top_DNN_derivative_norm_plot = top_DNN_derivative_norm_plot(:);
figure(str2num([num2str(hidden_number),'6']))
semilogy([1:length(RK4GRUcell_derivative_norm_plot)],RK4GRUcell_derivative_norm_plot,'k-');
xlim([0,length(RK4GRUcell_derivative_norm_plot)]);
% ylim([0,1e2]);
xlabel('Number of iterations');
ylabel('RK4GRUcell derivative norm');
set(gca,'fontsize',12);
figure(str2num([num2str(hidden_number),'7']))
semilogy([1:length(top_DNN_derivative_norm_plot)],top_DNN_derivative_norm_plot,'b-.');
xlim([0,length(top_DNN_derivative_norm_plot)]);
% ylim([0,1e2]);
xlabel('Epoch');
ylabel('top DNN derivative norm');
set(gca,'fontsize',12);


%%%%%%%%%%%%%%%%%Predicted error%%%%%%%%%%%%%%%%%%%%%%%

%%% all predicted samples
error_mse = mean(mean((E_X_train - E_X_pred).^2))^0.5;
error_mse_relative = error_mse/mean(mean((E_X_train).^2))^0.5;

error_max = max(max(abs(E_X_train - E_X_pred)));
error_max_relative = error_max/max(max(abs(E_X_train)));

error_loss = mean(mean((E_X_train - E_X_pred).^loss_L))^(1/loss_L);
error_loss_relative = error_mse/mean(mean((E_X_train).^loss_L))^(1/loss_L);

disp('  ')
disp(['Predicted error mse: ',num2str(error_mse)]);
disp(['Predicted error mse relative: ',num2str(error_mse_relative)])
disp(['Predicted error max: ',num2str(error_max)]);
disp(['Predicted error max relative: ',num2str(error_max_relative)])
disp(['Predicted error loss: ',num2str(error_loss)]);
disp(['Predicted error loss relative: ',num2str(error_loss_relative)])


%%% training samples
[~,num_sample] = size(E_X_train);
num_sample_select = 10000; % 训练整序列个数
dnum_sample = num_sample/num_sample_select;

index = [1:dnum_sample:num_sample];
indexl = index([5:5:num_sample_select]);
indexll = setdiff(index,indexl);

error_mse_train = mean(mean((E_X_train(:,indexll) - E_X_pred(:,indexll)).^2))^0.5;
error_mse_relative_train = error_mse_train/mean(mean((E_X_train).^2))^0.5;

error_max_train = max(max(abs(E_X_train(:,indexll) - E_X_pred(:,indexll))));
error_max_relative_train = error_max_train/max(max(abs(E_X_train)));

disp('  ')
disp(['Predicted error mse train: ',num2str(error_mse_train)]);
disp(['Predicted error mse relative train: ',num2str(error_mse_relative_train)]);
disp(['Predicted error max train: ',num2str(error_max_train)]);
disp(['Predicted error max relative train: ',num2str(error_max_relative_train)]);