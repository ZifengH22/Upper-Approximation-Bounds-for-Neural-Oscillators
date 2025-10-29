clear;clc;
% close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'Acc.mat']);
load([Data_path,'X1_response.mat']);
load([Data_path,'E_X1_response.mat']);

X_l = X1;
E_X_l = E_X1;

[num_time,num_sample] = size(X_l);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%resorting data
% [~,indx] = sort(std(Acc),'descend');
[~,indx] = sort(E_X_l(end,:)','descend');

Acc_train = Acc(:,indx);
X_train = X_l(:,indx);
E_X_train = E_X_l(:,indx);

%%%%%%%%%%%%%%
num_time_select = 1000;  % 训练序列长度
num_sample_select_initial = 27869; % 训练整序列个数 11569 for 5000 E_X 27869 for 10000 E_X

%%%selecting index%%%
index_select = zeros(num_sample_select_initial,1);
index_all = [1:1:num_sample]';
% value_select = std(Acc_train)';
value_select = E_X_train(end,:)';

value_select_range = value_select(1) - value_select(end);
value_select_interval = value_select_range/(num_sample_select_initial-1);
for i = 1:num_sample_select_initial
    [~, index_temp] = min(abs(value_select - (value_select(1) - value_select_interval*(i-1)) ));
    index_select(i) = index_all(index_temp);
end
index_select = unique(index_select);
num_sample_select = length(index_select);

%%%%%%selecting training data%%%%%%%%%%
F_train = zeros(num_sample_select,num_time_select,1);
X_dX_input_train = zeros(num_sample_select,1,2);
X_dX_output_train = zeros(num_sample_select,num_time_select,1);
E_X_output_train = zeros(num_sample_select,num_time_select,1);

for i = 1:num_sample_select
    F_l = Acc_train(:, index_select(i));
    F_train(i,:,:) = F_l;
    X_dX_l = X_train(:,index_select(i));
    X_dX_output_train(i,:,:) = X_dX_l;
    E_X_l = E_X_train(:,index_select(i));
    E_X_output_train(i,:,:) = E_X_l;
end

% figure(1)
% plot(value_select(index_select),'-o')
% figure(2)
% plot(E_X_train(end,index_select) - E_X_output_train(:,end)')
% figure(3)
% hist(E_X_output_train(:,end));

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
save([Data_path,'index_select.mat'],'index_select');
save([Data_path,'coef_F.mat'],'coef_F');
save([Data_path,'coef_X_output.mat'],'coef_X_output');
save([Data_path,'coef_dX_output.mat'],'coef_dX_output');
save([Data_path,'coef_ddX_output.mat'],'coef_ddX_output');
save([Data_path,'coef_E_X_output.mat'],'coef_E_X_output');
save([Data_path,'index_select.mat'],'index_select');

