clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'Acc.mat']);
load([Data_path,'X1_response.mat']);
load([Data_path,'E_X1_response.mat']);
% load([Data_path,'dX1_response.mat']);
% load([Data_path,'ddX1_response.mat']);

X_l = X1;
E_X_l = E_X1;

[num_time,num_sample] = size(X_l);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,indx] = sort(std(Acc),'descend');
% [~,indx] = sort(std(Acc));

num_delay_F = 0;
num_delay_X = 0;

Acc_train = [zeros(num_delay_F,num_sample);Acc(:,indx)];
X_train = [zeros(num_delay_X,num_sample);X_l(:,indx)];
E_X_train = [zeros(num_delay_X,num_sample);E_X_l(:,indx)];

num_time_select = 1000;  % 训练序列长度
num_sample_select = 2000; % 训练整序列个数
dnum_sample = num_sample/num_sample_select;

F_train = zeros(num_sample_select,num_time_select,1);
X_dX_input_train = zeros(num_sample_select,1,2);
X_dX_output_train = zeros(num_sample_select,num_time_select,1);
E_X_output_train = zeros(num_sample_select,num_time_select,1);

for i = 1:num_sample_select
    F_l = Acc_train(:, (i-1)*dnum_sample+1);
    F_train(i,:,:) = F_l;
    X_dX_l = X_train(:,(i-1)*dnum_sample+1);
    X_dX_output_train(i,:,:) = X_dX_l;
    E_X_l = E_X_train(:,(i-1)*dnum_sample+1);
    E_X_output_train(i,:,:) = E_X_l;
end

%%%%%
index = [1:1:num_sample_select];
indexl = [5:5:num_sample_select];
indexll = setdiff(index,indexl);

F_l = F_train(indexl,:,:);
F_ll = F_train(indexll,:,:);
F_train = [F_ll;F_l];

X_dX_input_train_l = X_dX_input_train(indexl,:,:);
X_dX_input_train_ll = X_dX_input_train(indexll,:,:);
X_dX_input_train = [X_dX_input_train_ll;X_dX_input_train_l];

X_dX_output_train_l = X_dX_output_train(indexl,:,:);
X_dX_output_train_ll = X_dX_output_train(indexll,:,:);
X_dX_output_train = [X_dX_output_train_ll;X_dX_output_train_l];

E_X_output_train_l = E_X_output_train(indexl,:,:);
E_X_output_train_ll = E_X_output_train(indexll,:,:);
E_X_output_train = [E_X_output_train_ll;E_X_output_train_l];

t_train = repmat(t,num_sample_select,1,1);
%%%%

X_output_train = squeeze(X_dX_output_train(:,:,1));
dX_output_train_app = diff(X_output_train')'/dt;
ddX_output_train_app = diff(dX_output_train_app')'/dt;

coef_F = std(F_train(:));
coef_X_output = std(X_output_train(:));
coef_dX_output = std(dX_output_train_app(:));
coef_ddX_output = std(ddX_output_train_app(:));
coef_E_X_output = std(E_X_output_train(:));

numTrain = size(X_dX_input_train,1); %总训练样本数

disp([' '])
disp(['The number of trained whole time series is: ',num2str(num_sample_select)])
disp([' '])
disp(['The length of each trained sample is: ',num2str(num_time_select)])
disp([' '])
disp(['The total number of trained samples is: ',num2str(numTrain)])

%%%%%save data%%%%%%
save([Data_path,'Acc_train.mat'],'Acc_train', '-v7.3');
save([Data_path,'X_train.mat'],'X_train', '-v7.3');
save([Data_path,'E_X_train.mat'],'E_X_train', '-v7.3');

save([Data_path,'t_train.mat'],'t_train', '-v7.3');
save([Data_path,'F_train.mat'],'F_train', '-v7.3');
save([Data_path,'X_dX_input_train.mat'],'X_dX_input_train', '-v7.3');
save([Data_path,'X_dX_output_train.mat'],'X_dX_output_train', '-v7.3');
save([Data_path,'E_X_output_train.mat'],'E_X_output_train', '-v7.3');

save([Data_path,'num_sample_select.mat'],'num_sample_select');
save([Data_path,'coef_F.mat'],'coef_F');
save([Data_path,'coef_X_output.mat'],'coef_X_output');
save([Data_path,'coef_dX_output.mat'],'coef_dX_output');
save([Data_path,'coef_ddX_output.mat'],'coef_ddX_output');
save([Data_path,'coef_E_X_output.mat'],'coef_E_X_output');