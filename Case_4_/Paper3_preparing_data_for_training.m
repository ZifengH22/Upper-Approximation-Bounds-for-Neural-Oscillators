clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'Acc.mat']);
load([Data_path,'X200_response.mat']);
[num_time,num_sample] = size(Acc);
dt = 0.005;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,indx] = sort(std(Acc),'descend');
Acc_sort = Acc(:,indx)';
X200_sort = X200(indx,:);


num_sample_select = 2000; 
dnum_sample = num_sample/num_sample_select;

Acc_train = zeros(num_sample_select,num_time,1);
X_dX_input_train = zeros(num_sample_select,1,2);
X_dX_output_train = zeros(num_sample_select,num_time,1);
X_dX_input_test = zeros(num_sample,1,2);

for i = 1:num_sample_select
    Acc_l = Acc_sort((i-1)*dnum_sample+1,:);
    Acc_train(i,:,:) = Acc_l;

    X200_l = X200_sort((i-1)*dnum_sample+1,:);
    X_dX_output_train(i,:,:) = X200_l;
end

%%%%%
index = [1:1:num_sample_select];
indexl = [5:5:num_sample_select];
indexll = setdiff(index,indexl);

Acc_l = Acc_train(indexl,:,:);
Acc_ll = Acc_train(indexll,:,:);
Acc_train = [Acc_ll;Acc_l];

X_dX_input_train_l = X_dX_input_train(indexl,:,:);
X_dX_input_train_ll = X_dX_input_train(indexll,:,:);
X_dX_input_train = [X_dX_input_train_ll;X_dX_input_train_l];

X_dX_output_train_l = X_dX_output_train(indexl,:,:);
X_dX_output_train_ll = X_dX_output_train(indexll,:,:);
X_dX_output_train = [X_dX_output_train_ll;X_dX_output_train_l];

t_train = repmat(t,num_sample_select,1,1);

%%%%%
numTrain = size(X_dX_output_train,1);

disp([' '])
disp(['The number of trained whole time series is: ',num2str(num_sample_select)])
disp([' '])
disp(['The length of each trained sample is: ',num2str(num_time)])
disp([' '])
disp(['The total number of trained samples is: ',num2str(numTrain)])

%%%%%save data%%%%%%
save([Data_path,'Acc_train.mat'],'Acc_train', '-v7.3');
save([Data_path,'X_dX_input_train.mat'],'X_dX_input_train', '-v7.3');
save([Data_path,'X_dX_output_train.mat'],'X_dX_output_train', '-v7.3');
save([Data_path,'X_dX_input_test.mat'],'X_dX_input_test', '-v7.3');
save([Data_path,'t_train.mat'],'t_train', '-v7.3');
save([Data_path,'num_sample_select.mat'],'num_sample_select');
